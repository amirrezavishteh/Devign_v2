"""Initial node representation x_v = concat(Code, Type) (Sec 2.2.2).

Code is the precomputed word2vec feature (passed in as code_feat). Type is a label-encoded node
type id, here embedded with a learnable nn.Embedding of size `type_dim`. Concatenated they form
the d = code_dim + type_dim dimensional x_v (d = 200 with the paper's 100+100 split).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NodeInitEmbedding(nn.Module):
    def __init__(self, code_dim: int, type_vocab_size: int, type_dim: int):
        super().__init__()
        self.code_dim = code_dim
        self.type_dim = type_dim
        self.out_dim = code_dim + type_dim
        self.type_embedding = nn.Embedding(type_vocab_size, type_dim, padding_idx=0)

    def forward(self, code_feat: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """code_feat [B, M, code_dim], type_ids [B, M] -> x [B, M, code_dim+type_dim]."""
        type_feat = self.type_embedding(type_ids)
        return torch.cat([code_feat, type_feat], dim=-1)
