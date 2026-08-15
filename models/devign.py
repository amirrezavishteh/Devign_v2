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
from models.conv_module import ConvModule, prior_logit
from models.ggnn import GatedGraphRecurrentLayer
from models.node_init import NodeInitEmbedding


class _Trunk(nn.Module):
    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int,
                 num_edge_types: int, hidden_dim: int, time_steps: int, aggregation: str,
                 type_init_std: float = 1.0):
        super().__init__()
        self.node_init = NodeInitEmbedding(code_dim, type_vocab_size, type_dim, type_init_std)
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
                 aggregation: str, conv_cfg: dict, mlp_hidden: int, dropout: float,
                 pos_rate: float | None = None, type_init_std: float = 1.0):
        super().__init__()
        self.trunk = _Trunk(code_dim, type_vocab_size, type_dim, num_edge_types,
                            hidden_dim, time_steps, aggregation, type_init_std)
        self.conv = ConvModule(hidden_dim, self.trunk.init_dim, conv_cfg, mlp_hidden, dropout,
                               pos_rate=pos_rate)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        H, x = self.trunk(batch)
        return self.conv(H, x, batch.mask)  # logits [B]


class GgrnModel(nn.Module):
    """Eq. 5: y~ = Sigmoid( SUM_j MLP([H_j, x_j]) ). Masked sum over real nodes only."""

    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int,
                 num_edge_types: int, hidden_dim: int, time_steps: int,
                 aggregation: str, mlp_hidden: int, dropout: float,
                 logit_affine: bool = True, pos_rate: float | None = None,
                 type_init_std: float = 1.0):
        super().__init__()
        self.trunk = _Trunk(code_dim, type_vocab_size, type_dim, num_edge_types,
                            hidden_dim, time_steps, aggregation, type_init_std)
        in_dim = hidden_dim + self.trunk.init_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        # Same graph-level affine the Conv module gets, so the Q2 comparison isolates the readout
        # (Conv vs flat summation) rather than which of the two was handed the class prior.
        #
        # `scale` stays at 1 here and is deliberately NOT given the Conv module's larger init:
        # Eq. 5 sums a per-node logit over every node, so its output is already O(M) rather than
        # the ~1e-3 a product of two near-zero heads produces. There is no attenuation to undo,
        # and amplifying a sum that already scales with graph size would only make it worse.
        self.logit_affine = logit_affine
        if logit_affine:
            self.scale = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.tensor([prior_logit(pos_rate)]))

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        H, x = self.trunk(batch)
        node_repr = torch.cat([H, x], dim=-1)              # [B, M, z+d]
        node_logits = self.mlp(node_repr).squeeze(-1)      # [B, M]
        node_logits = node_logits * batch.mask.float()
        logits = node_logits.sum(dim=1)                     # [B]
        if self.logit_affine:
            logits = self.scale * logits + self.bias
        return logits


def build_model(name: str, cfg: dict, code_dim: int, type_vocab_size: int,
                num_edge_types: int, pos_rate: float | None = None) -> nn.Module:
    """`pos_rate` is the TRAINING split's positive fraction; it initialises the readout bias so
    the model does not have to learn the class prior through Eq. 9's product. Pass None (or set
    `model.conv.logit_affine: false`) for the paper's exact readout."""
    m = cfg["model"]
    emb = cfg["embedding"]
    if name == "devign":
        return DevignModel(
            code_dim=code_dim, type_vocab_size=type_vocab_size, type_dim=emb["type_dim"],
            num_edge_types=num_edge_types, hidden_dim=m["hidden_dim"],
            time_steps=m["time_steps"], aggregation=m["aggregation"],
            conv_cfg=m["conv"], mlp_hidden=m["conv"]["mlp_hidden"], dropout=m["dropout"],
            pos_rate=pos_rate, type_init_std=emb.get("type_init_std", 1.0),
        )
    if name == "ggrn":
        return GgrnModel(
            code_dim=code_dim, type_vocab_size=type_vocab_size, type_dim=emb["type_dim"],
            num_edge_types=num_edge_types, hidden_dim=m["hidden_dim"],
            time_steps=m["time_steps"], aggregation=m["aggregation"],
            mlp_hidden=m["conv"]["mlp_hidden"], dropout=m["dropout"],
            logit_affine=bool(m["conv"].get("logit_affine", True)), pos_rate=pos_rate,
            type_init_std=emb.get("type_init_std", 1.0),
        )
    raise ValueError(f"unknown model: {name}")
