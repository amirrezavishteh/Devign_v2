"""Gated attention MIL pooling as a drop-in alternative to the Conv module (Eq. 6-9).

Following Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance Learning" (ICML 2018):

    e_j   = w^T ( tanh(V h_j) (elemwise*) sigmoid(U h_j) )     # gated attention, per node
    a     = softmax(e over REAL nodes only)                     # padded nodes -> -inf first
    z     = SUM_j a_j h_j                                       # bag embedding
    logit = Linear(z)                                           # single scalar

Why this belongs here
---------------------
The Conv module's stated job (Sec 2.4) is to "select sets of nodes and features that are relevant
to the current graph-level task". That is a description of attention-based multiple-instance
pooling, and Eq. 9 is a degenerate implementation of it: it ends in a product of two near-zero MLP
heads, which starves the trunk through both paths at once (measured at 384x weaker than a linear
control on real data -- see scripts/measure_readout.py).

MIL pooling is linear in the node embeddings, so its gradient is alive at initialisation with no
rescue term. That is hypothesis H3, and it is checked directly rather than assumed.

The localisation that falls out is not a bonus feature bolted on: `a` is a distribution over
nodes, every node carries a source span, so a per-line score is a projection of `a` rather than a
separate model. The paper listed interpretability as future work.

Masking is not optional
-----------------------
Graphs are padded to the batch's max node count. A softmax over the padded canvas would let a
graph's prediction depend on which graphs it was batched with -- a silent, batch-order-dependent
bug that no unit test of a single graph would catch. Padded logits are set to -inf before the
softmax, so they receive exactly zero attention mass.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedAttentionPool(nn.Module):
    """Attention MIL pooling over graph nodes.

    `heads` > 1 runs k independent attention distributions and concatenates their bag embeddings
    (Ilse et al. Sec 2.4's multi-head variant). k=1 is the headline configuration; k>1 is an
    ablation, because more heads buy capacity and blur the "which node mattered" story that
    localisation depends on.
    """

    def __init__(self, in_dim: int, attn_dim: int = 128, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        if heads < 1:
            raise ValueError(f"mil_heads must be >= 1, got {heads}")
        self.heads = heads
        self.attn_dim = attn_dim

        # V and U are the two gated branches; w projects their product to one score per head.
        self.V = nn.Linear(in_dim, attn_dim * heads)
        self.U = nn.Linear(in_dim, attn_dim * heads)
        self.w = nn.Parameter(torch.empty(heads, attn_dim))
        nn.init.xavier_uniform_(self.w)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim * heads, 1)

    def attention(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h [B, M, D], mask [B, M] bool -> attention [B, heads, M], rows summing to 1.

        Padded positions are exactly zero, and a graph's distribution is independent of its
        batch-mates.
        """
        B, M, _ = h.shape
        v = torch.tanh(self.V(h)).view(B, M, self.heads, self.attn_dim)
        u = torch.sigmoid(self.U(h)).view(B, M, self.heads, self.attn_dim)
        gated = v * u                                              # [B, M, heads, attn_dim]
        # einsum over the attention dim: one score per (graph, node, head).
        scores = torch.einsum("bmha,ha->bhm", gated, self.w)        # [B, heads, M]

        # -inf on padding, so softmax assigns it exactly zero mass. Using the dtype's min rather
        # than float('-inf') keeps this finite under autocast.
        pad = ~mask.unsqueeze(1)                                    # [B, 1, M]
        scores = scores.masked_fill(pad, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        # A graph with no real nodes would softmax over all -inf and produce NaN. Cannot happen
        # via the collate path, but a zero row is the right answer if it ever does.
        return attn.masked_fill(pad, 0.0)

    def forward(self, h: torch.Tensor, mask: torch.Tensor,
                return_attention: bool = False):
        """h [B, M, D], mask [B, M] -> logits [B] (and attention [B, heads, M] if asked)."""
        attn = self.attention(h, mask)                              # [B, heads, M]
        bag = torch.einsum("bhm,bmd->bhd", attn, h)                 # [B, heads, D]
        bag = bag.reshape(bag.shape[0], -1)                         # [B, heads*D]
        logits = self.classifier(self.dropout(bag)).squeeze(-1)     # [B]
        return (logits, attn) if return_attention else logits
