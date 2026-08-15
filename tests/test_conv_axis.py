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


def _cfg(axis: str, readout: str | None = None) -> dict:
    cfg = copy.deepcopy(load_config("config.yaml")["model"]["conv"])
    cfg["conv_axis"] = axis
    cfg["conv_channels"] = 8
    cfg["mlp_hidden"] = 12
    # avg_nodes needs a node *sequence*, which only the nodes axis produces.
    cfg["readout"] = readout or ("avg_nodes" if axis == "nodes" else "max_nodes")
    return cfg


@pytest.mark.parametrize("axis,readout", [
    ("nodes", "avg_nodes"), ("nodes", "max_nodes"), ("features", "max_nodes"),
])
@pytest.mark.parametrize("num_nodes", [1, 2, 5, 40])
def test_forward_and_backward_for_both_axes(axis, readout, num_nodes):
    """Must produce one logit per graph and a usable gradient, including tiny graphs."""
    torch.manual_seed(0)
    mod = ConvModule(hidden_dim=Z, init_dim=D, conv_cfg=_cfg(axis, readout), mlp_hidden=12,
                     dropout=0.0)
    H = torch.randn(B, num_nodes, Z, requires_grad=True)
    x = torch.randn(B, num_nodes, D)
    mask = torch.ones(B, num_nodes, dtype=torch.bool)

    logits = mod(H, x, mask)
    assert logits.shape == (B,)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()


def test_avg_nodes_readout_rejects_the_features_axis():
    """The two knobs are coupled: on the features axis the node axis is a spatial dimension of a
    4-D map, not a sequence position, so there is nothing for avg_nodes to average over."""
    with pytest.raises(ValueError, match="conv_axis"):
        ConvModule(hidden_dim=Z, init_dim=D, conv_cfg=_cfg("features", "avg_nodes"),
                   mlp_hidden=12, dropout=0.0)


@pytest.mark.parametrize("readout", ["avg_nodes", "max_nodes"])
def test_padded_nodes_do_not_change_a_graphs_logit(readout):
    """A graph's prediction must not depend on how large its batch-mates are.

    This is the invariant that keeps avg_nodes honest: the Linear heads have biases, so a padded
    position produces a NONZERO head output and would drag the average toward a constant if it
    were not masked out.
    """
    torch.manual_seed(0)
    mod = ConvModule(hidden_dim=Z, init_dim=D, conv_cfg=_cfg("nodes", readout), mlp_hidden=12,
                     dropout=0.0)
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


def test_avg_readout_is_sensitive_to_how_many_nodes_fire():
    """The point of AVG over max: doubling the number of high-activation nodes must move the
    logit. A global max cannot see that difference, which is why Eq. 9 averages."""
    torch.manual_seed(0)
    mod = ConvModule(hidden_dim=Z, init_dim=D, conv_cfg=_cfg("nodes", "avg_nodes"),
                     mlp_hidden=12, dropout=0.0)
    mod.eval()

    m = 40
    quiet = torch.zeros(1, m, Z)
    x = torch.zeros(1, m, D)
    mask = torch.ones(1, m, dtype=torch.bool)

    few, many = quiet.clone(), quiet.clone()
    spike = torch.randn(Z) * 5.0
    few[0, :4] = spike
    many[0, :20] = spike
    with torch.no_grad():
        assert not torch.allclose(mod(few, x, mask), mod(many, x, mask), atol=1e-5)


def test_node_axis_and_avg_readout_are_the_defaults():
    conv = load_config("config.yaml")["model"]["conv"]
    assert conv["conv_axis"] == "nodes"
    assert conv["readout"] == "avg_nodes"
