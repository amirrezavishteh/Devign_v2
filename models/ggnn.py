"""Gated Graph Recurrent Layer (Sec 2.3).

Implements, in batched dense form:

    h^(1)_j = [x_j^T, 0]^T                      # pad annotation x_j (dim d) up to z=hidden_dim
    a^(t-1)_{j,p} = A_p^T ( W_p H^(t-1) + b )    # per edge-type message  (Eq. 3)
    AGG = SUM_p a^(t-1)_{j,p}                    # SUM aggregation over k edge types (Eq. 4)
    h^(t)_j = GRU(h^(t-1)_j, AGG)                # GRU update           (Eq. 4)

repeated for T time steps, returning H^(T) of shape [B, M, z].

A_p is the dense (optionally row-normalized) adjacency for edge type p. A_p^T (W_p H) means: each
node j gathers the transformed states of its in-neighbors under edge type p. Padded nodes are
zeroed via `mask` so they neither send nor receive messages.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedGraphRecurrentLayer(nn.Module):
    def __init__(self, num_edge_types: int, hidden_dim: int, time_steps: int,
                 aggregation: str = "sum"):
        super().__init__()
        self.k = num_edge_types
        self.z = hidden_dim
        self.T = time_steps
        self.aggregation = aggregation.lower()
        assert self.aggregation in {"sum", "mean", "max", "concat"}

        # One linear transform W_p (+ bias b) per edge type. Implemented as a single fused linear
        # of out-features = k * z so all k messages are computed in one matmul.
        self.edge_transform = nn.Linear(hidden_dim, num_edge_types * hidden_dim)

        gru_in = hidden_dim * self.k if self.aggregation == "concat" else hidden_dim
        self.gru = nn.GRUCell(gru_in, hidden_dim)

    def _aggregate(self, messages: torch.Tensor) -> torch.Tensor:
        # messages: [B, k, M, z]
        if self.aggregation == "sum":
            return messages.sum(dim=1)
        if self.aggregation == "mean":
            return messages.mean(dim=1)
        if self.aggregation == "max":
            return messages.max(dim=1).values
        # concat
        B, k, M, z = messages.shape
        return messages.permute(0, 2, 1, 3).reshape(B, M, k * z)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None, mask: torch.Tensor,
                edge_index: torch.Tensor | None = None,
                edge_type: torch.Tensor | None = None,
                edge_norm: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, M, d] (d <= z), mask [B, M] -> H^(T) [B, M, z].

        Supply either `adj` [B, k, M, M] (dense) or `edge_index`/`edge_type` (sparse). The two
        paths are numerically identical; the sparse one is O(E*z) instead of O(k*M^2) and is what
        makes real 500-node graphs fit at batch_size 128.
        """
        if edge_index is None and adj is None:
            raise ValueError("GatedGraphRecurrentLayer needs either adj or edge_index")
        B, M, d = x.shape
        device = x.device

        # h^(1) = [x^T, 0]^T : copy annotation into the first d dims, zero-pad to z.
        h = torch.zeros(B, M, self.z, device=device, dtype=x.dtype)
        h[:, :, :d] = x
        mask_f = mask.float().unsqueeze(-1)  # [B, M, 1]
        h = h * mask_f

        for _ in range(self.T):
            if edge_index is not None:
                agg = self._propagate_sparse(h, edge_index, edge_type, edge_norm, B, M)
            else:
                # W_p H + b for all p:  [B, M, k*z] -> [B, k, M, z]
                transformed = self.edge_transform(h).view(B, M, self.k, self.z).permute(0, 2, 1, 3)
                # A_p^T (W_p H): for each edge type, aggregate over source nodes.
                # adj[b,p] is [M(src), M(dst)] with 1 at (s,t) meaning s->t; A_p^T gathers on src.
                messages = torch.matmul(adj.transpose(-1, -2), transformed)  # [B, k, M, z]
                agg = self._aggregate(messages)  # [B, M, z] (or [B, M, k*z] for concat)

            # GRU update per node (flatten batch*nodes into GRUCell's batch dim).
            agg_flat = agg.reshape(B * M, -1)
            h_flat = h.reshape(B * M, self.z)
            h_new = self.gru(agg_flat, h_flat).view(B, M, self.z)
            h = h_new * mask_f  # keep padded states at zero

        return h

    def _propagate_sparse(self, h: torch.Tensor, edge_index: torch.Tensor,
                          edge_type: torch.Tensor, edge_norm: torch.Tensor | None,
                          B: int, M: int) -> torch.Tensor:
        """Eq. 3 + Eq. 4's SUM, fused over an edge list instead of k dense adjacencies."""
        N = B * M
        # W_p h_src + b for every p, then keep only each edge's own type slice.
        transformed = self.edge_transform(h).view(N, self.k, self.z)   # [N, k, z]
        src, dst = edge_index[0], edge_index[1]
        msg = transformed[src, edge_type]                              # [E, z]
        if edge_norm is not None:
            msg = msg * edge_norm.unsqueeze(-1)

        if self.aggregation == "concat":
            # Keep messages separated per edge type, then flatten to [N, k*z].
            out = h.new_zeros(N, self.k, self.z)
            out.index_put_((dst, edge_type), msg, accumulate=True)
            return out.view(B, M, self.k * self.z)

        if self.aggregation in {"sum", "mean"}:
            out = h.new_zeros(N, self.z).index_add_(0, dst, msg)
            if self.aggregation == "mean":
                # Mean over the k edge types, matching the dense path (which averages the k
                # per-type message matrices, including the types with no incoming edge).
                out = out / self.k
            return out.view(B, M, self.z)

        # max: aggregate per (node, edge type) first, then take the max across types.
        per_type = h.new_zeros(N, self.k, self.z)
        per_type.index_put_((dst, edge_type), msg, accumulate=True)
        return per_type.max(dim=1).values.view(B, M, self.z)
