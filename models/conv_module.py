"""The Conv module (Sec 2.4, Eqs. 6-9).

Two parallel branches:
    Z^(1) = sigma([H^(T), x]),  Z^(2) = sigma(Z^(1))          # branch over concat(H, x)
    Y^(1) = sigma(H^(T)),       Y^(2) = sigma(Y^(1))          # branch over H alone
    y~ = Sigmoid( AVG( MLP(Z^(l)) (.) MLP(Y^(l)) ) )          # (.) = elementwise product

where sigma(.) = MAXPOOL(ReLU(CONV(.))) (Eq. 6) and l=2 conv layers.

Layout
------
Each graph's node-embedding matrix is a 2-D map [nodes (M), features (W)] with 1 input channel,
so the paper's filters map onto Conv2d/MaxPool2d with (height, width) = (nodes, features).
Sec 3.3 specifies, and this module now uses:

    conv1 (1, 3) + ReLU   ->  pool1 (1, 3) filter, (1, 2) stride
    conv2 (1, 1)          ->  pool2 (2, 2) filter, (1, 2) stride

Note the asymmetry that matters: conv1/pool1/conv2 all have height 1, so they act purely along
the feature axis. Only **pool2 has height 2**, which is where the module mixes information
between *adjacent* nodes. That single filter is the paper's node-level interaction, so getting it
right is not cosmetic.

Padding and batch-invariance
----------------------------
Graphs are padded to the per-batch max node count M, so a naive pool over the padded canvas would
make a graph's prediction depend on its batch-mates. Two guards keep the readout invariant:

  1. Before pool2 (the only node-mixing pool) padded rows are set to -inf, so a real node at the
     end of a graph never max-pools with a padded neighbour.
  2. The node axis is finally collapsed by a masked max-pool over real rows only.

With pool2's (1, 2) stride the node axis shrinks by exactly one row (H_out = M - 1), and since
padding is right-aligned, output row i is real iff input row i is real -- so the final mask is
just `mask[:, :H_out]`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MLP_OUT = 16  # common dimension D of MLP(Z) and MLP(Y) before the elementwise product

# Sec 3.3 defaults, used when config.yaml omits a key.
_DEFAULTS = {
    "conv1_filter": (1, 3),
    "pool1_filter": (1, 3),
    "pool1_stride": (1, 2),
    "conv2_filter": (1, 1),
    "pool2_filter": (2, 2),
    "pool2_stride": (1, 2),
}


def _pair(cfg: dict, key: str) -> tuple[int, int]:
    v = cfg.get(key) or _DEFAULTS[key]
    return int(v[0]), int(v[1])


def _neg_inf_pad(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set padded node rows to -inf so a max-pool never mixes them into a real node."""
    m = mask.view(mask.shape[0], 1, mask.shape[1], 1)          # [B,1,M,1]
    return x.masked_fill(~m, torch.finfo(x.dtype).min)


class _SigmaStack(nn.Module):
    """Two CONV->ReLU->MAXPOOL layers (Eq. 6/7) over a [B, 1, M, W] map.

    Filters come from config (`model.conv.*`), defaulting to the paper's Sec 3.3 values.
    """

    def __init__(self, conv_channels: int, in_width: int, conv_cfg: dict):
        super().__init__()
        c1 = _pair(conv_cfg, "conv1_filter")
        p1, p1s = _pair(conv_cfg, "pool1_filter"), _pair(conv_cfg, "pool1_stride")
        c2 = _pair(conv_cfg, "conv2_filter")
        p2, p2s = _pair(conv_cfg, "pool2_filter"), _pair(conv_cfg, "pool2_stride")

        # Pad only the feature axis, so the node axis is never inflated with synthetic rows.
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=c1, padding=(0, c1[1] // 2))
        self.pool1 = nn.MaxPool2d(kernel_size=p1, stride=p1s, ceil_mode=True)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=c2,
                               padding=(0, c2[1] // 2))
        self.pool2 = nn.MaxPool2d(kernel_size=p2, stride=p2s, ceil_mode=True)

        self.pool2_height = p2[0]
        self.pool2_stride_h = p2s[0]
        self.conv_channels = conv_channels
        self.out_width = self._infer_out_width(in_width)

    def _infer_out_width(self, in_width: int) -> int:
        with torch.no_grad():
            # Height must clear every height-kernel; pool2's is the only one > 1.
            h = max(4, self.pool2_height + 1)
            dummy = torch.zeros(1, 1, h, in_width)
            out = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy))))))
        return out.shape[-1]

    def node_out_height(self, m: int) -> int:
        """Rows remaining on the node axis after pool2, for a map of `m` input rows."""
        import math
        return max(1, math.ceil((m - self.pool2_height) / self.pool2_stride_h) + 1)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = feat.unsqueeze(1)                    # [B, 1, M, W]
        x = self.pool1(F.relu(self.conv1(x)))    # height untouched (kernel/stride 1)
        x = F.relu(self.conv2(x))
        if self.pool2_height > 1:
            x = _neg_inf_pad(x, mask)            # keep padding out of the node-mixing pool
        x = self.pool2(x)                        # [B, C, H_out, W']
        return x


def _masked_node_maxpool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """feat [B, C, H, W'], mask [B, M] -> [B, C, W'] : max over real nodes only."""
    h = feat.shape[2]
    m = mask[:, :h].view(mask.shape[0], 1, h, 1)             # [B,1,H,1]
    neg = torch.finfo(feat.dtype).min
    masked = feat.masked_fill(~m, neg)
    pooled = masked.max(dim=2).values                        # [B, C, W']
    # Guard against a (degenerate) all-padded row: replace the sentinel with 0.
    pooled = torch.where(pooled <= neg / 2, torch.zeros_like(pooled), pooled)
    return pooled


class ConvModule(nn.Module):
    def __init__(self, hidden_dim: int, init_dim: int, conv_cfg: dict, mlp_hidden: int,
                 dropout: float = 0.2):
        super().__init__()
        ch = conv_cfg["conv_channels"]
        zx_width = hidden_dim + init_dim   # width of [H, x]
        y_width = hidden_dim               # width of H
        self.branch_zx = _SigmaStack(ch, zx_width, conv_cfg)
        self.branch_y = _SigmaStack(ch, y_width, conv_cfg)
        self.dropout = nn.Dropout(dropout)

        flat_zx = ch * self.branch_zx.out_width
        flat_y = ch * self.branch_y.out_width
        self.mlp_zx = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_zx, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, MLP_OUT))
        self.mlp_y = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_y, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, MLP_OUT))

    def forward(self, H: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """H [B, M, z], x [B, M, d], mask [B, M] -> logits [B] (pre-sigmoid)."""
        # pool2 has a height-2 kernel, so a 1-node graph has no adjacent node to pool with.
        if H.shape[1] < self.branch_zx.pool2_height:
            pad = self.branch_zx.pool2_height - H.shape[1]
            H = F.pad(H, (0, 0, 0, pad))
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)

        zx_in = torch.cat([H, x], dim=-1)                 # [B, M, z+d]
        Z = self.branch_zx(zx_in, mask)                   # [B, C, H_out, W'_zx]
        Y = self.branch_y(H, mask)                        # [B, C, H_out, W'_y]

        Z = _masked_node_maxpool(Z, mask)                 # [B, C, W'_zx]
        Y = _masked_node_maxpool(Y, mask)                 # [B, C, W'_y]
        Z = self.dropout(Z)
        Y = self.dropout(Y)

        z_out = self.mlp_zx(Z)   # [B, MLP_OUT]
        y_out = self.mlp_y(Y)    # [B, MLP_OUT]
        prod = z_out * y_out     # elementwise multiply (Eq. 9)
        logits = prod.mean(dim=-1)  # AVG aggregation -> [B]
        return logits
