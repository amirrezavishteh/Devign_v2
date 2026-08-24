"""Gated attention MIL pooling: correctness of the masking, and parity with the other readouts.

The load-bearing test here is batch invariance. Graphs are padded to the batch's max node count,
so a softmax that includes padded positions would make a graph's prediction depend on which
graphs it happened to be batched with. That bug produces no error, no NaN and no failing shape --
it just makes every result quietly irreproducible and batch-order dependent. It is exactly the
class of defect this project exists to stop shipping, so it is asserted rather than reasoned about.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from devign_data.dataset import GraphSample, dedupe_edges, make_collate_fn
from devign_data.graph_builder import EDGE_TYPES
from models.devign import build_model
from models.mil_pool import GatedAttentionPool
from training.utils import load_config

CODE_DIM = 16
IN_DIM = 24


def _sample(rng, m, label=0):
    edges = {et: dedupe_edges(rng.integers(0, m, size=(2, 2 * m)).astype(np.int64))
             for et in EDGE_TYPES}
    return GraphSample(
        code_feat=rng.normal(size=(m, CODE_DIM)).astype(np.float32),
        type_ids=rng.integers(0, 6, size=m).astype(np.int64),
        edges=edges, num_nodes=m, label=label, project="p")


@pytest.fixture
def pool():
    torch.manual_seed(0)
    return GatedAttentionPool(IN_DIM, attn_dim=8, heads=1).eval()


def test_attention_sums_to_one_over_real_nodes(pool):
    h = torch.randn(3, 10, IN_DIM)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :10] = True
    mask[2, :1] = True
    attn = pool.attention(h, mask)
    assert torch.allclose(attn.sum(-1), torch.ones(3, 1), atol=1e-6)


def test_padded_nodes_receive_exactly_zero_attention(pool):
    h = torch.randn(2, 12, IN_DIM)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :9] = True
    attn = pool.attention(h, mask)
    assert attn[0, :, 5:].abs().max() == 0.0
    assert attn[1, :, 9:].abs().max() == 0.0


def test_padding_content_cannot_change_the_logit(pool):
    """Whatever garbage sits in the padded rows must not reach the output."""
    torch.manual_seed(1)
    h = torch.randn(2, 8, IN_DIM)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :5] = True

    noisy = h.clone()
    noisy[:, 5:, :] = 1e4          # extreme values in the padding
    with torch.no_grad():
        a = pool(h, mask)
        b = pool(noisy, mask)
    assert torch.equal(a, b), "padded content leaked into the graph logit"


def test_graph_logit_is_independent_of_its_batch_mates(pool):
    """The same graph, padded to two different widths, must give the identical logit."""
    torch.manual_seed(2)
    g = torch.randn(1, 6, IN_DIM)

    narrow_mask = torch.ones(1, 6, dtype=torch.bool)
    wide = torch.cat([g, torch.randn(1, 20, IN_DIM)], dim=1)     # padded out to 26 nodes
    wide_mask = torch.zeros(1, 26, dtype=torch.bool)
    wide_mask[0, :6] = True

    with torch.no_grad():
        narrow_logit = pool(g, narrow_mask)
        wide_logit = pool(wide, wide_mask)
        narrow_attn = pool.attention(g, narrow_mask)
        wide_attn = pool.attention(wide, wide_mask)

    assert torch.allclose(narrow_logit, wide_logit, atol=1e-6), (
        f"logit moved with batch width: {narrow_logit.item():.6f} vs {wide_logit.item():.6f}")
    assert torch.allclose(narrow_attn, wide_attn[:, :, :6], atol=1e-6), (
        "attention distribution moved with batch width")


@pytest.mark.parametrize("heads", [1, 4])
def test_multi_head_shapes_and_normalisation(heads):
    torch.manual_seed(3)
    p = GatedAttentionPool(IN_DIM, attn_dim=8, heads=heads).eval()
    h = torch.randn(2, 7, IN_DIM)
    mask = torch.ones(2, 7, dtype=torch.bool)
    logits, attn = p(h, mask, return_attention=True)
    assert logits.shape == (2,)
    assert attn.shape == (2, heads, 7)
    assert torch.allclose(attn.sum(-1), torch.ones(2, heads), atol=1e-6)


def test_mil_shares_the_trunk_and_matches_devign_in_size():
    """The three-arm comparison is only meaningful if the readout is the only thing that differs."""
    cfg = load_config("config.yaml")
    kw = dict(code_dim=100, type_vocab_size=50, num_edge_types=7, pos_rate=0.43)
    mil = build_model("mil", cfg, **kw)
    devign = build_model("devign", cfg, **kw)

    # Same trunk architecture, parameter for parameter.
    mil_trunk = {n: p.shape for n, p in mil.trunk.named_parameters()}
    devign_trunk = {n: p.shape for n, p in devign.trunk.named_parameters()}
    assert mil_trunk == devign_trunk

    # Total size within 5%, so neither arm wins on capacity alone.
    n_mil = sum(p.numel() for p in mil.parameters())
    n_devign = sum(p.numel() for p in devign.parameters())
    assert abs(n_mil - n_devign) / n_devign < 0.05, (n_mil, n_devign)


def test_mil_forward_and_backward_on_a_real_batch():
    cfg = load_config("config.yaml")
    cfg["embedding"].update(word2vec_dim=CODE_DIM, type_dim=CODE_DIM)
    cfg["model"].update(hidden_dim=32, time_steps=2)
    rng = np.random.default_rng(0)
    batch = make_collate_fn(EDGE_TYPES, sparse=True)(
        [_sample(rng, 9, 1), _sample(rng, 15, 0), _sample(rng, 4, 1)])

    model = build_model("mil", cfg, code_dim=CODE_DIM, type_vocab_size=8,
                        num_edge_types=len(EDGE_TYPES), pos_rate=0.5)
    logits, attn = model(batch, return_attention=True)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    # Attention must respect each graph's true node count.
    assert attn[0, :, 9:].abs().max() == 0.0
    assert attn[2, :, 4:].abs().max() == 0.0

    logits.sum().backward()
    grad = sum(float(p.grad.pow(2).sum()) for p in model.trunk.parameters()
               if p.grad is not None) ** 0.5
    assert grad > 0.0, "no gradient reached the GGNN trunk"
