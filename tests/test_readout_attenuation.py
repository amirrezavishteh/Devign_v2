"""Lock in the Eq. 9 initialisation pathology as an executable assertion.

The paper's readout ends in a product of two MLP heads (Eq. 9). Both start near zero, and because
d(z*y)/dz = y and d(z*y)/dy = z, each branch's gradient is scaled by the other branch's smallness
-- so the trunk is starved through both paths simultaneously. These tests assert the consequence
rather than the mechanism: at initialisation the conv readout must pass a far weaker gradient to
the GGNN trunk than a plain linear head on the same pooled features does.

The thresholds are deliberately loose. The exact ratio depends on width, batch and seed (measured
between ~50x and ~170x across configurations), so asserting a precise value would make this a
change-detector rather than a test. What is being defended is the ORDER OF MAGNITUDE and the
DIRECTION, which is what the claim in the write-up actually rests on.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from devign_data.dataset import GraphSample, dedupe_edges, make_collate_fn
from devign_data.graph_builder import EDGE_TYPES
from scripts.measure_readout import measure
from training.utils import load_config

HIDDEN = 64
CODE_DIM = 32


@pytest.fixture(scope="module")
def cfg():
    c = load_config("config.yaml")
    c["project"].update(seed=42, device="cpu")
    c["model"].update(hidden_dim=HIDDEN, time_steps=3, dropout=0.0)
    c["embedding"].update(word2vec_dim=CODE_DIM, type_dim=CODE_DIM)
    c["model"]["conv"] = dict(c["model"]["conv"], conv_channels=16, mlp_hidden=16)
    return c


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    samples = []
    for i in range(16):
        m = int(rng.integers(20, 40))
        edges = {et: dedupe_edges(rng.integers(0, m, size=(2, 2 * m)).astype(np.int64))
                 for et in EDGE_TYPES}
        samples.append(GraphSample(
            code_feat=rng.normal(size=(m, CODE_DIM)).astype(np.float32),
            type_ids=rng.integers(0, 8, size=m).astype(np.int64),
            edges=edges, num_nodes=m, label=int(i % 2), project="p"))
    return make_collate_fn(EDGE_TYPES, sparse=True)(samples)


@pytest.fixture(scope="module")
def rows(cfg, batch):
    return {r["readout"]: r for r in measure(cfg, "cpu", batch, seed=42)}


def test_paper_readout_is_degenerate_at_init(rows):
    """Eq. 9 as written: every probability lands on top of 0.5."""
    off = rows["conv_affine_off"]
    assert off["prob_spread"] < 0.01, (
        f"expected a collapsed probability band, got {off['prob_spread']:.3e}")
    assert 0.49 < off["prob_min"] and off["prob_max"] < 0.51


def test_conv_readout_starves_the_trunk_relative_to_a_linear_head(rows):
    """The measurement the Eq. 9 claim rests on."""
    off = rows["conv_affine_off"]
    linear = rows["linear_head"]
    ratio = linear["trunk_grad_norm"] / off["trunk_grad_norm"]
    assert ratio > 10.0, (
        f"conv readout passed {ratio:.1f}x less gradient than a linear head; the reported "
        f"pathology is an order-of-magnitude effect, so anything under 10x means the claim "
        f"needs re-measuring, not restating")


def test_logit_affine_recovers_most_of_the_lost_gradient(rows):
    """The repo's fix must actually move the number it was added to move."""
    off, on = rows["conv_affine_off"], rows["conv_affine_on"]
    assert on["trunk_grad_norm"] > off["trunk_grad_norm"] * 5, (
        "logit_affine did not meaningfully restore the trunk gradient")
    assert on["prob_spread"] > off["prob_spread"] * 5


def test_flat_summation_was_never_starved(rows):
    """Eq. 5 sums a per-node logit over every node, so its output is O(M), not O(1e-3).

    This is why the Ggrn baseline trains from epoch 1 while Devign spends its first epochs
    escaping the dead zone -- the paper's own ablation arm does not share the defect.
    """
    ggrn = rows["ggrn_sum"]
    assert ggrn["prob_spread"] > 0.05
    assert ggrn["trunk_grad_norm"] > rows["conv_affine_off"]["trunk_grad_norm"] * 100


def test_every_variant_actually_reached_the_trunk(rows):
    """A zero gradient anywhere would mean the measurement is broken, not that the readout is."""
    for name, r in rows.items():
        assert r["trunk_grad_norm"] > 0.0, f"{name} passed no gradient to the trunk at all"
        assert np.isfinite(r["trunk_grad_norm"]), f"{name} produced a non-finite gradient"
