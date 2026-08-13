"""Full Devign model and the Ggrn baseline (flat weighted summation, Eq. 5).

Devign  : NodeInit -> GatedGraphRecurrentLayer -> ConvModule -> sigmoid
Ggrn    : NodeInit -> GatedGraphRecurrentLayer -> SUM(MLP([H, x])) -> sigmoid   (Eq. 5)

Both share the embedding + GGNN trunk; they differ only in the readout, which is exactly the
ablation the paper studies for Q2 ("Conv module vs flat summation").
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data.dataset import GraphBatch
from models.conv_module import ConvModule
from models.ggnn import GatedGraphRecurrentLayer
from models.node_init import NodeInitEmbedding


class _Trunk(nn.Module):
    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int,
                 num_edge_types: int, hidden_dim: int, time_steps: int, aggregation: str):
        super().__init__()
        self.node_init = NodeInitEmbedding(code_dim, type_vocab_size, type_dim)
        self.init_dim = self.node_init.out_dim
        assert hidden_dim >= self.init_dim, "hidden_dim (z) must be >= annotation dim d"
        self.ggnn = GatedGraphRecurrentLayer(num_edge_types, hidden_dim, time_steps, aggregation)

    def forward(self, batch: GraphBatch):
        x = self.node_init(batch.code_feat, batch.type_ids)   # [B, M, d]
        x = x * batch.mask.float().unsqueeze(-1)
        H = self.ggnn(x, batch.adj, batch.mask,               # [B, M, z]
                      edge_index=batch.edge_index, edge_type=batch.edge_type,
                      edge_norm=batch.edge_norm)
        return H, x


class DevignModel(nn.Module):
    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int,
                 num_edge_types: int, hidden_dim: int, time_steps: int,
                 aggregation: str, conv_cfg: dict, mlp_hidden: int, dropout: float):
        super().__init__()
        self.trunk = _Trunk(code_dim, type_vocab_size, type_dim, num_edge_types,
                            hidden_dim, time_steps, aggregation)
        self.conv = ConvModule(hidden_dim, self.trunk.init_dim, conv_cfg, mlp_hidden, dropout)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        H, x = self.trunk(batch)
        return self.conv(H, x, batch.mask)  # logits [B]


class GgrnModel(nn.Module):
    """Eq. 5: y~ = Sigmoid( SUM_j MLP([H_j, x_j]) ). Masked sum over real nodes only."""

    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int,
                 num_edge_types: int, hidden_dim: int, time_steps: int,
                 aggregation: str, mlp_hidden: int, dropout: float):
        super().__init__()
        self.trunk = _Trunk(code_dim, type_vocab_size, type_dim, num_edge_types,
                            hidden_dim, time_steps, aggregation)
        in_dim = hidden_dim + self.trunk.init_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        H, x = self.trunk(batch)
        node_repr = torch.cat([H, x], dim=-1)              # [B, M, z+d]
        node_logits = self.mlp(node_repr).squeeze(-1)      # [B, M]
        node_logits = node_logits * batch.mask.float()
        return node_logits.sum(dim=1)                       # [B]


def build_model(name: str, cfg: dict, code_dim: int, type_vocab_size: int,
                num_edge_types: int) -> nn.Module:
    m = cfg["model"]
    emb = cfg["embedding"]
    if name == "devign":
        return DevignModel(
            code_dim=code_dim, type_vocab_size=type_vocab_size, type_dim=emb["type_dim"],
            num_edge_types=num_edge_types, hidden_dim=m["hidden_dim"],
            time_steps=m["time_steps"], aggregation=m["aggregation"],
            conv_cfg=m["conv"], mlp_hidden=m["conv"]["mlp_hidden"], dropout=m["dropout"],
        )
    if name == "ggrn":
        return GgrnModel(
            code_dim=code_dim, type_vocab_size=type_vocab_size, type_dim=emb["type_dim"],
            num_edge_types=num_edge_types, hidden_dim=m["hidden_dim"],
            time_steps=m["time_steps"], aggregation=m["aggregation"],
            mlp_hidden=m["conv"]["mlp_hidden"], dropout=m["dropout"],
        )
    raise ValueError(f"unknown model: {name}")
