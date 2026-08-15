"""The Conv module's convolution axis (Sec 2.4, "1-D convolutional layers").

The original implementation slid a width-3 kernel along the node embedding's FEATURE axis. That
is almost certainly wrong: word2vec dimensions are unordered, so dimensions 5 and 6 are no more
related than 5 and 87, and sharing convolution weights across them is strictly worse than a dense
layer. The node axis, by contrast, carries real order (AST pre-order / natural code sequence), so
a width-3 kernel over it sees three consecutive code elements.

Both orientations are kept so they can be compared; these tests pin the shapes and the default.
"""
from __future__ import annotations

import copy

import pytest
import torch

from models.conv_module import ConvModule
from training.utils import load_config

Z, D, B = 24, 16, 3


def _cfg(axis: str) -> dict:
    cfg = copy.deepcopy(load_config("config.yaml")["model"]["conv"])
    cfg["conv_axis"] = axis
    cfg["conv_channels"] = 8
    cfg["mlp_hidden"] = 12
    return cfg


@pytest.mark.parametrize("axis", ["nodes", "features"])
@pytest.mark.parametrize("num_nodes", [1, 2, 5, 40])
def test_forward_and_backward_for_both_axes(axis, num_nodes):
    """Must produce one logit per graph and a usable gradient, including tiny graphs."""
    torch.manual_seed(0)
    mod = ConvModule(hidden_dim=Z, init_dim=D, conv_cfg=_cfg(axis), mlp_hidden=12, dropout=0.0)
    H = torch.randn(B, num_nodes, Z, requires_grad=True)
    x = torch.randn(B, num_nodes, D)
    mask = torch.ones(B, num_nodes, dtype=torch.bool)

    logits = mod(H, x, mask)
    assert logits.shape == (B,)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()


def test_padded_nodes_do_not_change_a_graphs_logit():
    """A graph's prediction must not depend on how large its batch-mates are."""
    torch.manual_seed(0)
    mod = ConvModule(hidden_dim=Z, init_dim=D, conv_cfg=_cfg("nodes"), mlp_hidden=12, dropout=0.0)
    mod.eval()

    m = 12
    H = torch.randn(1, m, Z)
    x = torch.randn(1, m, D)
    mask = torch.ones(1, m, dtype=torch.bool)
    with torch.no_grad():
        tight = mod(H, x, mask)

    pad = 20
    Hp = torch.cat([H, torch.randn(1, pad, Z)], dim=1)
    xp = torch.cat([x, torch.randn(1, pad, D)], dim=1)
    maskp = torch.cat([mask, torch.zeros(1, pad, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        padded = mod(Hp, xp, maskp)

    assert torch.allclose(tight, padded, atol=1e-4), (
        f"padding changed the logit: {tight.item():.6f} vs {padded.item():.6f}")


def test_node_axis_is_the_default():
    assert load_config("config.yaml")["model"]["conv"]["conv_axis"] == "nodes"
