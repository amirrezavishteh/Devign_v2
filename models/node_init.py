"""Initial node representation x_v = concat(Code, Type) (Sec 2.2.2).

Code is the precomputed word2vec feature (passed in as code_feat). Type is a label-encoded node
type id, here embedded with a learnable nn.Embedding of size `type_dim`. Concatenated they form
the d = code_dim + type_dim dimensional x_v (d = 200 with the paper's 100+100 split).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NodeInitEmbedding(nn.Module):
    """x_v = concat(Code, Type).

    `type_init_std` sets the std of the Type embedding's initialisation, which decides how the two
    halves of x_v are balanced at the start of training. The halves are only comparable if their
    per-dimension scales are:

      * `embedding.internal_code: mean_standardized` gives the Code half unit per-dimension
        variance, so the default 1.0 (also PyTorch's `nn.Embedding` default) matches it.
      * `zero` / `mean` leave raw word2vec vectors, whose per-dimension std is more like 0.2-0.6.
        Against those, a std-1.0 Type embedding is several times larger and dominates x_v -- and
        therefore h^(1) -- for the whole early phase of training. Lower this to match.
    """

    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int,
                 type_init_std: float = 1.0):
        super().__init__()
        self.code_dim = code_dim
        self.type_dim = type_dim
        self.out_dim = code_dim + type_dim
        self.type_embedding = nn.Embedding(type_vocab_size, type_dim, padding_idx=0)
        if type_init_std != 1.0:
            nn.init.normal_(self.type_embedding.weight, mean=0.0, std=type_init_std)
            with torch.no_grad():
                self.type_embedding.weight[0].zero_()   # restore padding_idx

    def forward(self, code_feat: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """code_feat [B, M, code_dim], type_ids [B, M] -> x [B, M, code_dim+type_dim]."""
        type_feat = self.type_embedding(type_ids)
        return torch.cat([code_feat, type_feat], dim=-1)
