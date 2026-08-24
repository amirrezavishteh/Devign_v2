"""Line spans and the attention -> line projection.

The line-span test matters more than it looks. Byte offsets are offsets into the UTF-8 ENCODING,
so counting newlines in the decoded `str` drifts by one line for every non-ASCII character earlier
in the file -- and C source does contain them, in comments and string literals. The drift is
silent and would misattribute every subsequent line.
"""
from __future__ import annotations

import numpy as np
import pytest

from devign_data.dataset import node_line_spans
from devign_data.graph_builder import build_graph
from evaluation.localize import (LocalizationUnavailable, attention_to_lines,
                                 length_prior_baseline, localization_metrics, rank_lines)

SRC = """int f(int *p, int n) {
    int i = 0;
    if (p == NULL) {
        return -1;
    }
    for (i = 0; i < n; i++) {
        p[i] = i;
    }
    return i;
}
"""


def test_line_spans_are_1_based_and_within_the_source():
    g = build_graph(SRC)
    spans = node_line_spans(g, SRC)
    n_lines = SRC.count("\n") + 1
    assert spans.shape == (g.num_nodes, 2)
    assert spans.min() >= 1
    assert spans.max() <= n_lines
    assert (spans[:, 1] >= spans[:, 0]).all(), "a node ended before it started"


def test_root_node_spans_the_whole_function():
    g = build_graph(SRC)
    spans = node_line_spans(g, SRC)
    assert spans[0][0] == 1
    assert spans[0][1] >= 9


def test_a_known_token_lands_on_its_actual_line():
    """`NULL` appears only on line 3; whichever node holds it must say line 3."""
    g = build_graph(SRC)
    spans = node_line_spans(g, SRC)
    hits = {int(spans[i][0]) for i, n in enumerate(g.nodes) if n.code == "NULL"}
    assert hits == {3}, f"expected NULL on line 3, got {hits}"


def test_non_ascii_earlier_in_the_file_does_not_shift_later_lines():
    """The bug this guards: counting newlines in the str rather than the encoded bytes."""
    src = '/* ééé你好 */\nint g(void) {\n    return 42;\n}\n'
    g = build_graph(src)
    spans = node_line_spans(g, src)
    # `42` is on line 3 no matter how many bytes the comment on line 1 occupies.
    hits = {int(spans[i][0]) for i, n in enumerate(g.nodes) if n.code == "42"}
    assert hits == {3}, f"non-ASCII shifted the line numbering: got {hits}"


def test_attention_projects_onto_covered_lines():
    node_lines = np.array([[1, 3], [2, 2], [5, 5]], dtype=np.int32)
    attn = np.array([0.2, 0.7, 0.1])
    mx = attention_to_lines(attn, node_lines, num_nodes=3, aggregation="max")
    assert mx[1] == pytest.approx(0.2)
    assert mx[2] == pytest.approx(0.7)     # max(0.2 from the span, 0.7 from the node)
    assert mx[3] == pytest.approx(0.2)
    assert mx[5] == pytest.approx(0.1)
    assert 4 not in mx                      # no node covers line 4

    sm = attention_to_lines(attn, node_lines, num_nodes=3, aggregation="sum")
    assert sm[2] == pytest.approx(0.9)      # 0.2 + 0.7


def test_multi_head_attention_is_averaged():
    node_lines = np.array([[1, 1], [2, 2]], dtype=np.int32)
    attn = np.array([[1.0, 0.0], [0.0, 1.0]])   # two heads disagreeing
    out = attention_to_lines(attn, node_lines, num_nodes=2)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.5)


def test_missing_line_spans_raise_rather_than_scoring_line_zero():
    with pytest.raises(LocalizationUnavailable):
        attention_to_lines(np.array([1.0]), None, num_nodes=1)


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError):
        attention_to_lines(np.array([1.0]), np.array([[1, 1]]), 1, aggregation="mean")


def test_metrics_on_a_perfect_and_a_worst_case_ranking():
    perfect = localization_metrics([7, 1, 2, 3], {7})
    assert perfect["top_1"] == 1.0 and perfect["mrr"] == 1.0 and perfect["ifa"] == 0.0

    worst = localization_metrics([1, 2, 3, 7], {7})
    assert worst["top_1"] == 0.0
    assert worst["ifa"] == 3.0                       # three clean lines read first
    assert worst["mrr"] == pytest.approx(0.25)
    assert worst["normalised_rank"] == pytest.approx(1.0)


def test_metrics_are_none_when_there_is_nothing_to_find():
    assert localization_metrics([1, 2, 3], set()) is None
    assert localization_metrics([], {1}) is None
    assert localization_metrics([1, 2], {99}) is None   # truth not among the ranked lines


def test_rank_lines_is_deterministic_under_ties():
    scores = {5: 0.5, 2: 0.5, 9: 0.5}
    assert rank_lines(scores) == [2, 5, 9]


def test_length_prior_ranks_longest_first():
    m = length_prior_baseline({1: 10, 2: 80, 3: 5}, {2})
    assert m["top_1"] == 1.0 and m["ifa"] == 0.0
