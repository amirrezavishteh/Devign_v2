"""Sanity tests for the Devign pipeline. Run with: python -m pytest tests/ -q"""
from __future__ import annotations

import numpy as np
import torch

from data.graph_builder import EDGE_TYPES, build_graph
from data.dataset import GraphSample, make_collate_fn
from data.word2vec_embed import (NodeFeaturizer, TypeVocab, build_corpus,
                                train_word2vec)
from models.devign import build_model
from training.utils import load_config

_EXAMPLE = """
short add(short b) {
    short a = 32767;
    if (b > 0) {
        a = a + b;
    }
    return a;
}
"""


def test_graph_builder_produces_all_edge_types():
    g = build_graph(_EXAMPLE)
    assert g is not None and g.num_nodes > 0
    for et in EDGE_TYPES:
        assert et in g.edges
    # AST edges == num_nodes - 1 for a single tree.
    assert len(g.edges["AST"]) == g.num_nodes - 1
    assert len(g.edges["REV_AST"]) == len(g.edges["AST"])
    assert len(g.edges["CFG"]) > 0
    assert len(g.edges["NCS"]) > 0
    # DFG_R/DFG_W/DFG_C were previously never asserted at all. `b` is read in the `if` condition
    # (LastRead), `a` is written twice (LastWrite chain), and `a = a + b` computes `a` from `a`
    # and `b` (ComputedFrom) -- so on this example every DFG relation must be non-empty.
    assert len(g.edges["DFG_R"]) > 0
    assert len(g.edges["DFG_W"]) > 0
    assert len(g.edges["DFG_C"]) > 0


def test_max_nodes_filter():
    assert build_graph(_EXAMPLE, max_nodes=3) is None


def test_parse_error_filter_rejects_grossly_broken_code():
    from data.graph_builder import build_graph as _bg
    malformed = "int broken(int a { return a +++ ; ]]] }"
    assert _bg(malformed) is None
    assert _bg(malformed, drop_parse_errors=False) is not None


def test_parse_filter_keeps_code_with_a_localised_error():
    """A macro or GNU extension tree-sitter flags locally must NOT cost the whole function.

    Rejecting on any ERROR node discarded a measured 12.1% of the real corpus (~3,300 functions).
    """
    from data.graph_builder import build_graph as _bg
    # Realistic FFmpeg/QEMU shape: unknown attribute macro, otherwise perfectly ordinary C.
    src = """
    static av_cold int decode_init(AVCodecContext *avctx) {
        MyCtx *s = avctx->priv_data;
        int i, ret = 0;
        for (i = 0; i < 16; i++) {
            s->buf[i] = i * 2;
        }
        if (ret < 0) return ret;
        return 0;
    }
    """
    g = _bg(src)
    assert g is not None, "a function with only localised parse noise was discarded"
    assert g.num_nodes > 20


def test_parse_info_reports_error_extent():
    from data.parser import flatten_ast
    nodes, _, info = flatten_ast("int ok(void) { return 1; }", return_error=True)
    assert nodes
    assert info.has_function_definition
    assert info.error_fraction == 0.0
    assert info.is_usable()

    _, _, bad = flatten_ast("]]] ++ ;;; @@@", return_error=True)
    assert not bad.is_usable()


def test_featurizer_dims():
    g = build_graph(_EXAMPLE)
    corpus = build_corpus([g])
    w2v = train_word2vec(corpus, dim=16, epochs=2, min_count=1)
    tv = TypeVocab()
    for n in g.nodes:
        tv.add(n.type)
    feat = NodeFeaturizer(w2v, tv)
    code, types = feat.featurize_graph(g)
    assert code.shape == (g.num_nodes, 16)
    assert types.shape == (g.num_nodes,)
    assert types.dtype == np.int64


def test_internal_nodes_carry_type_only_by_default():
    """Leaves keep lexical content; internal nodes must not be pre-loaded with subtree means.

    Mean-pooling made root vectors from different functions 0.834-cosine-similar on real data,
    i.e. the upper AST carried almost no discriminative signal. Aggregation is the GGNN's job.
    """
    g = build_graph(_EXAMPLE)
    corpus = build_corpus([g])
    w2v = train_word2vec(corpus, dim=16, epochs=2, min_count=1)
    tv = TypeVocab()
    for n in g.nodes:
        tv.add(n.type)

    code, types = NodeFeaturizer(w2v, tv).featurize_graph(g)
    is_leaf = np.array([n.is_leaf for n in g.nodes])
    assert np.allclose(code[~is_leaf], 0.0), "internal nodes should carry no code vector"
    assert np.abs(code[is_leaf]).sum() > 0, "leaf nodes lost their token vectors"
    # Type still identifies every node, including internal ones.
    assert types.shape == (g.num_nodes,)
    assert len(set(types[~is_leaf].tolist())) > 1

    # The legacy behaviour remains available for comparison.
    mean_code, _ = NodeFeaturizer(w2v, tv, internal_code="mean").featurize_graph(g)
    assert np.abs(mean_code[~is_leaf]).sum() > 0


def test_featurizer_roundtrips_internal_code_mode(tmp_path):
    """Inference must featurize exactly as training did, or it scores a different function."""
    g = build_graph(_EXAMPLE)
    w2v = train_word2vec(build_corpus([g]), dim=16, epochs=2, min_count=1)
    tv = TypeVocab()
    for n in g.nodes:
        tv.add(n.type)

    out = str(tmp_path / "featurizer")
    NodeFeaturizer(w2v, tv, internal_code="mean").save(out)
    assert NodeFeaturizer.load(out).internal_code == "mean"

    NodeFeaturizer(w2v, tv, internal_code="zero").save(out)
    assert NodeFeaturizer.load(out).internal_code == "zero"


def test_model_forward_and_backward():
    cfg = load_config("config.yaml")
    g = build_graph(_EXAMPLE)
    corpus = build_corpus([g])
    w2v = train_word2vec(corpus, dim=cfg["embedding"]["word2vec_dim"], epochs=2, min_count=1)
    tv = TypeVocab()
    for n in g.nodes:
        tv.add(n.type)
    feat = NodeFeaturizer(w2v, tv)
    code, types = feat.featurize_graph(g)
    edges = {et: (np.array(g.edges[et], dtype=np.int64).T if g.edges[et]
                  else np.zeros((2, 0), dtype=np.int64)) for et in EDGE_TYPES}
    samples = [GraphSample(code, types, edges, g.num_nodes, label=1, project="t"),
               GraphSample(code, types, edges, g.num_nodes, label=0, project="t")]
    batch = make_collate_fn(EDGE_TYPES)(samples)

    for name in ("devign", "ggrn"):
        model = build_model(name, cfg, code_dim=cfg["embedding"]["word2vec_dim"],
                            type_vocab_size=len(tv), num_edge_types=len(EDGE_TYPES))
        logits = model(batch)
        assert logits.shape == (2,)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


if __name__ == "__main__":
    test_graph_builder_produces_all_edge_types()
    test_max_nodes_filter()
    test_featurizer_dims()
    test_model_forward_and_backward()
    print("all tests passed")
