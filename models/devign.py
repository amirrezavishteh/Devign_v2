"""Full Devign model and the Ggrn baseline (flat weighted summation, Eq. 5).

Devign  : NodeInit -> GatedGraphRecurrentLayer -> ConvModule -> sigmoid          (Eq. 6-9)
Ggrn    : NodeInit -> GatedGraphRecurrentLayer -> SUM(MLP([H, x])) -> sigmoid    (Eq. 5)
Mil     : NodeInit -> GatedGraphRecurrentLayer -> gated attention pooling -> sigmoid

All three share the embedding + GGNN trunk and differ only in the readout. The first two are the
ablation the paper studies for Q2 ("Conv module vs flat summation"); the third turns that binary
into a real comparison against an operator whose gradient is not degenerate at initialisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from devign_data.dataset import GraphBatch
from models.conv_module import ConvModule, prior_logit
from models.ggnn import GatedGraphRecurrentLayer
from models.mil_pool import GatedAttentionPool
from models.node_init import NodeInitEmbedding


class _Trunk(nn.Module):
    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int,
                 num_edge_types: int, hidden_dim: int, time_steps: int, aggregation: str,
                 type_init_std: float = 1.0, fast: bool = False):
        super().__init__()
        self.node_init = NodeInitEmbedding(code_dim, type_vocab_size, type_dim, type_init_std)
        self.init_dim = self.node_init.out_dim
        assert hidden_dim >= self.init_dim, "hidden_dim (z) must be >= annotation dim d"
        self.ggnn = GatedGraphRecurrentLayer(num_edge_types, hidden_dim, time_steps, aggregation)
        # Atomic (non-deterministic) message accumulation. Faster, and unusable for any reported
        # number -- see GatedGraphRecurrentLayer._propagate_sparse.
        self.fast = fast

    def forward(self, batch: GraphBatch):
        x = self.node_init(batch.code_feat, batch.type_ids)   # [B, M, d]
        x = x * batch.mask.float().unsqueeze(-1)
        H = self.ggnn(x, batch.adj, batch.mask,               # [B, M, z]
                      edge_index=batch.edge_index, edge_type=batch.edge_type,
                      edge_norm=batch.edge_norm,
                      seg_lengths_dst=batch.seg_lengths_dst,
                      seg_lengths_dst_type=batch.seg_lengths_dst_type,
                      fast=self.fast)
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


class MilModel(nn.Module):
    """NodeInit -> GGNN trunk -> gated attention MIL pooling -> sigmoid.

    Identical trunk to DevignModel and GgrnModel, so a three-way comparison isolates the readout:
    `conv` (Eq. 9), `sum` (Eq. 5) and `mil` differ in nothing else. The attention input is
    concat(H, x) -- the same [H, x] the Conv module's Z branch consumes -- for the same reason.

    No logit affine. That is the point of H3: MIL pooling is linear in the node embeddings, so it
    should have a live gradient at initialisation with no rescue term. Adding one here would make
    the hypothesis untestable.
    """

    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int,
                 num_edge_types: int, hidden_dim: int, time_steps: int,
                 aggregation: str, dropout: float, attn_dim: int = 128, heads: int = 1,
                 type_init_std: float = 1.0):
        super().__init__()
        self.trunk = _Trunk(code_dim, type_vocab_size, type_dim, num_edge_types,
                            hidden_dim, time_steps, aggregation, type_init_std)
        self.pool = GatedAttentionPool(hidden_dim + self.trunk.init_dim,
                                       attn_dim=attn_dim, heads=heads, dropout=dropout)

    def forward(self, batch: GraphBatch, return_attention: bool = False):
        H, x = self.trunk(batch)
        node_repr = torch.cat([H, x], dim=-1)              # [B, M, z+d]
        return self.pool(node_repr, batch.mask, return_attention=return_attention)

    @torch.no_grad()
    def attention(self, batch: GraphBatch) -> torch.Tensor:
        """Per-node attention [B, heads, M] for localisation. Padded nodes are exactly 0."""
        H, x = self.trunk(batch)
        return self.pool.attention(torch.cat([H, x], dim=-1), batch.mask)


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
        #
        # That is correct as far as it goes, and it misses the opposite failure. Measured at
        # initialisation on a real 128-graph batch (scripts/measure_readout.py), Eq. 5 does not
        # merely avoid Eq. 9's dead start -- it SATURATES: logits land in [+1.8, +36.9], every
        # probability is >= 0.86 with the maximum pinned at 1.0, and the loss is 11.01 against
        # ln(2) = 0.693. The model begins by calling every graph vulnerable with near-total
        # confidence on a split that is 43.2% positive.
        #
        # So both of the paper's readouts are badly conditioned at init, in opposite directions:
        # Eq. 9 collapses onto 0.5, Eq. 5 blows through the top of the sigmoid. That matters for
        # the paper's own Q2 ("does the Conv module beat flat summation?") because the comparison
        # is between two poorly-initialised readouts, not between a good one and a bad one. The
        # fix is NOT to raise `scale`; it would be to normalise the sum by node count, which is a
        # deviation from Eq. 5 as written and is therefore left off by default and recorded here.
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
    if name == "mil":
        conv = m["conv"]
        return MilModel(
            code_dim=code_dim, type_vocab_size=type_vocab_size, type_dim=emb["type_dim"],
            num_edge_types=num_edge_types, hidden_dim=m["hidden_dim"],
            time_steps=m["time_steps"], aggregation=m["aggregation"], dropout=m["dropout"],
            attn_dim=int(m.get("mil_attn_dim", 128)), heads=int(m.get("mil_heads", 1)),
            type_init_std=emb.get("type_init_std", 1.0),
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
