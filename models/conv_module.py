"""The Conv module (Sec 2.4, Eqs. 6-9).

Two parallel branches:
    Z^(1) = sigma([H^(T), x]),  Z^(2) = sigma(Z^(1))          # branch over concat(H, x)
    Y^(1) = sigma(H^(T)),       Y^(2) = sigma(Y^(1))          # branch over H alone
    y~ = Sigmoid( AVG( MLP(Z^(l)) (.) MLP(Y^(l)) ) )          # (.) = elementwise product

where sigma(.) = MAXPOOL(ReLU(CONV(.))) (Eq. 6) and l=2 conv layers. The AVG in Eq. 9 runs over
node positions -- see `ConvModule.readout` for that and for the alternative.

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

import math

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
    """Two CONV->ReLU->MAXPOOL layers (Eq. 6/7).

    Two orientations are supported, selected by `model.conv.conv_axis`:

    ``"nodes"`` (default)
        A genuine **1-D convolution over the node sequence**, with the node's feature vector as
        input channels: `Conv1d(W -> C, kernel 3)` sliding along nodes, then max-pooling along
        nodes. This reads the paper's "1-D convolutional layers" (Sec 2.4) as sliding over the
        code, which is the only axis with meaningful order (AST pre-order / natural code
        sequence), so a width-3 kernel sees three consecutive code elements.

    ``"features"``
        The original orientation: a (1,3) kernel sliding along the **feature** axis of the node
        embedding. This is almost certainly wrong -- word2vec dimensions are unordered, so
        dimensions 5 and 6 are no more related than 5 and 87, and weight-sharing across them is
        strictly worse than a dense layer. Kept only so the two can be compared.
    """

    def __init__(self, conv_channels: int, in_width: int, conv_cfg: dict):
        super().__init__()
        self.axis = str(conv_cfg.get("conv_axis", "nodes")).lower()
        c1 = _pair(conv_cfg, "conv1_filter")
        p1, p1s = _pair(conv_cfg, "pool1_filter"), _pair(conv_cfg, "pool1_stride")
        c2 = _pair(conv_cfg, "conv2_filter")
        p2, p2s = _pair(conv_cfg, "pool2_filter"), _pair(conv_cfg, "pool2_stride")
        self.conv_channels = conv_channels

        if self.axis == "nodes":
            # Widths of the paper's filters become kernel sizes along the node axis.
            self.conv1 = nn.Conv1d(in_width, conv_channels, kernel_size=c1[1],
                                   padding=c1[1] // 2)
            self.pool1 = nn.MaxPool1d(kernel_size=p1[1], stride=p1s[1], ceil_mode=True)
            self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=c2[1],
                                   padding=c2[1] // 2)
            self.pool2 = nn.MaxPool1d(kernel_size=p2[1], stride=p2s[1], ceil_mode=True)
            self.out_width = conv_channels
        else:
            # Pad only the feature axis, so the node axis is never inflated with synthetic rows.
            self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=c1, padding=(0, c1[1] // 2))
            self.pool1 = nn.MaxPool2d(kernel_size=p1, stride=p1s, ceil_mode=True)
            self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=c2,
                                   padding=(0, c2[1] // 2))
            self.pool2 = nn.MaxPool2d(kernel_size=p2, stride=p2s, ceil_mode=True)
            self.pool2_height = p2[0]
            self.pool2_stride_h = p2s[0]
            self.out_width = self._infer_out_width(in_width)

    def _infer_out_width(self, in_width: int) -> int:
        with torch.no_grad():
            # Height must clear every height-kernel; pool2's is the only one > 1.
            h = max(4, self.pool2_height + 1)
            dummy = torch.zeros(1, 1, h, in_width)
            out = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy))))))
        return int(out.shape[-1])

    @property
    def min_nodes(self) -> int:
        """Smallest node count this stack can consume without a zero-size dimension."""
        return 4 if self.axis == "nodes" else max(2, getattr(self, "pool2_height", 1))

    @property
    def readout_dim(self) -> int:
        """Flat width the MLP sees after the node axis is collapsed by the max-pool."""
        # nodes: [B, C, M'] --max over M'--> [B, C]
        # features: [B, C, H, W'] --masked max over H--> [B, C, W'] --flatten--> C*W'
        return self.conv_channels if self.axis == "nodes" else self.conv_channels * self.out_width

    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        """Returns (activations, pooled_mask). `pooled_mask` is [B, 1, M'] for the ``nodes`` axis
        (node validity at the post-pool resolution, needed by the avg_nodes readout) and None for
        the ``features`` axis, where the node axis survives as a spatial dimension instead."""
        if self.axis == "nodes":
            # [B, M, W] -> [B, W, M]: features are channels, the node sequence is the signal.
            x = feat.transpose(1, 2)
            m = mask.unsqueeze(1).to(x.dtype)             # [B, 1, M]
            x = x * m

            # Re-zero padded positions after every conv. Convolution bias makes an all-padding
            # window produce a non-zero activation, and a window centred on the first pad node
            # still sees a real neighbour -- so without this a graph's logit would depend on how
            # long its batch-mates are. Everything here is post-ReLU and therefore non-negative,
            # so zeroing a position is exactly equivalent to excluding it from the max-pool.
            x = F.relu(self.conv1(x)) * m                 # conv1 padding keeps length == M
            x = self.pool1(x)
            m = (self.pool1(m) > 0).to(x.dtype)           # validity at the pooled resolution
            x = F.relu(self.conv2(x)) * m                 # conv2 kernel 1 keeps length
            x = self.pool2(x)
            m = (self.pool2(m) > 0).to(x.dtype)
            return x * m, m                               # [B, C, M'], pads exactly 0

        x = feat.unsqueeze(1)                    # [B, 1, M, W]
        x = self.pool1(F.relu(self.conv1(x)))    # height untouched (kernel/stride 1)
        x = F.relu(self.conv2(x))
        if self.pool2_height > 1:
            x = _neg_inf_pad(x, mask)            # keep padding out of the node-mixing pool
        x = self.pool2(x)                        # [B, C, H_out, W']
        return x, None


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


def prior_logit(pos_rate: float | None) -> float:
    """log(p / (1-p)) for the training positive rate, clamped away from the asymptotes."""
    if not pos_rate:
        return 0.0
    p = min(max(float(pos_rate), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


class ConvModule(nn.Module):
    def __init__(self, hidden_dim: int, init_dim: int, conv_cfg: dict, mlp_hidden: int,
                 dropout: float = 0.2, pos_rate: float | None = None):
        super().__init__()
        ch = conv_cfg["conv_channels"]
        zx_width = hidden_dim + init_dim   # width of [H, x]
        y_width = hidden_dim               # width of H
        self.branch_zx = _SigmaStack(ch, zx_width, conv_cfg)
        self.branch_y = _SigmaStack(ch, y_width, conv_cfg)
        self.dropout = nn.Dropout(dropout)

        # How the node axis is collapsed into a graph-level prediction.
        #
        #   "avg_nodes" (default) -- Eq. 9 as written: apply the MLPs at every surviving node
        #       position, multiply the two branches elementwise, and AVG over real positions. The
        #       average is what makes the readout sensitive to *how many* sites look vulnerable.
        #   "max_nodes" -- collapse the node axis with a global max BEFORE the MLPs, so the graph
        #       is summarised by one C-dim "did any window fire this filter" vector. That answers
        #       a different question (any vs how many) and is length-biased, since a longer
        #       function gets more chances to trigger a high max.
        self.readout = str(conv_cfg.get("readout", "avg_nodes")).lower()
        if self.readout not in {"avg_nodes", "max_nodes"}:
            raise ValueError(f"conv.readout must be avg_nodes|max_nodes, got {self.readout!r}")
        if self.readout == "avg_nodes" and self.branch_zx.axis != "nodes":
            raise ValueError(
                "conv.readout: avg_nodes requires conv.conv_axis: nodes -- on the features axis "
                "the node axis is a spatial dimension of a 4-D map, not a sequence position.")

        if self.readout == "avg_nodes":
            # Applied per node position, so no Flatten: nn.Linear maps the last (channel) dim of
            # [B, M', C]. Output stays [B, M', MLP_OUT] and is averaged in `forward`.
            def _head(in_dim: int) -> nn.Module:
                return nn.Sequential(nn.Linear(in_dim, mlp_hidden), nn.ReLU(),
                                     nn.Linear(mlp_hidden, MLP_OUT))
            self.mlp_zx = _head(self.branch_zx.conv_channels)
            self.mlp_y = _head(self.branch_y.conv_channels)
        else:
            def _head(in_dim: int) -> nn.Module:
                return nn.Sequential(nn.Flatten(), nn.Linear(in_dim, mlp_hidden), nn.ReLU(),
                                     nn.Linear(mlp_hidden, MLP_OUT))
            self.mlp_zx = _head(self.branch_zx.readout_dim)
            self.mlp_y = _head(self.branch_y.readout_dim)

        # Learnable affine on the graph-level logit. Eq. 9 has neither term, and both are needed
        # because it ends in a product of two MLP heads: at init both factors are near zero, so
        # the whole batch's logits span ~6e-4 (every probability lands in 0.5000-0.5008).
        #
        #   `bias`  starts at the training split's prior logit, so the model does not spend its
        #           first ~10 epochs discovering the class balance through a multiplicative
        #           bottleneck (observed: f1 exactly 0.00 for 8 epochs).
        #   `scale` multiplies the gradient flowing back into both branches. Measured, the
        #           gradient reaching the GGNN trunk through this head is ~53x weaker than through
        #           a linear head on the same pooled features, so `logit_scale_init` > 1 buys that
        #           magnitude back. Left at 1.0 by default -- a free hyperparameter, not a paper
        #           value.
        #
        # Set `conv.logit_affine: false` to recover the paper's exact readout.
        self.logit_affine = bool(conv_cfg.get("logit_affine", True))
        if self.logit_affine:
            self.scale = nn.Parameter(
                torch.tensor([float(conv_cfg.get("logit_scale_init", 1.0))]))
            self.bias = nn.Parameter(torch.tensor([prior_logit(pos_rate)]))

    def forward(self, H: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """H [B, M, z], x [B, M, d], mask [B, M] -> logits [B] (pre-sigmoid)."""
        # Graphs shorter than the stack's kernels have no window to convolve over.
        need = self.branch_zx.min_nodes
        if H.shape[1] < need:
            pad = need - H.shape[1]
            H = F.pad(H, (0, 0, 0, pad))
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)

        zx_in = torch.cat([H, x], dim=-1)                 # [B, M, z+d]
        Z, z_mask = self.branch_zx(zx_in, mask)
        Y, _ = self.branch_y(H, mask)

        if self.readout == "avg_nodes":
            # [B, C, M'] -> [B, M', C] so the heads act per node position.
            z_out = self.mlp_zx(self.dropout(Z.transpose(1, 2)))   # [B, M', MLP_OUT]
            y_out = self.mlp_y(self.dropout(Y.transpose(1, 2)))    # [B, M', MLP_OUT]
            prod = z_out * y_out                                   # Eq. 9's elementwise product
            # AVG over REAL positions only. Padded positions carry a nonzero head output (the
            # Linear biases fire on a zero input), so they must be excluded explicitly here --
            # unlike the max readout, where post-ReLU zeros can never win.
            keep = z_mask.transpose(1, 2)                          # [B, M', 1]
            denom = keep.sum(dim=(1, 2)).clamp(min=1.0) * prod.shape[-1]
            return self._affine((prod * keep).sum(dim=(1, 2)) / denom)

        if self.branch_zx.axis == "nodes":
            # [B, C, M'] -> max over the (pooled) node axis: "did any window fire this filter".
            # Padded positions are exactly 0 and every activation is post-ReLU (>= 0), so they
            # can only win when no real window fired at all -- which is the correct answer.
            Z = Z.max(dim=2).values
            Y = Y.max(dim=2).values
        else:
            Z = _masked_node_maxpool(Z, mask)             # [B, C, W'_zx]
            Y = _masked_node_maxpool(Y, mask)             # [B, C, W'_y]

        z_out = self.mlp_zx(self.dropout(Z))   # [B, MLP_OUT]
        y_out = self.mlp_y(self.dropout(Y))    # [B, MLP_OUT]
        prod = z_out * y_out                   # elementwise multiply (Eq. 9)
        return self._affine(prod.mean(dim=-1))  # AVG aggregation -> [B]

    def _affine(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale * logits + self.bias if self.logit_affine else logits
