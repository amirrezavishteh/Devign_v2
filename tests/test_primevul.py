"""Deriving vulnerable line numbers by diffing a function against its patch.

The derivation is the ground truth every localisation number rests on, so its edge cases are
asserted rather than assumed. Getting it subtly wrong -- off by one, counting insertions, counting
reindentation -- would not fail loudly; it would just quietly move every score.
"""
from __future__ import annotations

import json

import pytest

from devign_data.primevul import (line_lengths, load_primevul_paired,
                                  vulnerable_lines_from_diff)

BEFORE = """int copy(char *dst, const char *src, int n) {
    int i;
    for (i = 0; i <= n; i++) {
        dst[i] = src[i];
    }
    return i;
}"""

AFTER = """int copy(char *dst, const char *src, int n) {
    int i;
    for (i = 0; i < n; i++) {
        dst[i] = src[i];
    }
    return i;
}"""


def test_replaced_line_is_the_vulnerable_one():
    """The off-by-one is on line 3 and nowhere else."""
    assert vulnerable_lines_from_diff(BEFORE, AFTER) == {3}


def test_deleted_line_is_labelled():
    before = "a();\nbad();\nc();"
    after = "a();\nc();"
    assert vulnerable_lines_from_diff(before, after) == {2}


def test_pure_insertion_labels_nothing():
    """An added bounds check has no counterpart line in `before` to attribute it to.

    Counting it would score the model against a line that does not exist in the input it saw.
    """
    before = "use(p);"
    after = "if (!p) return -1;\nuse(p);"
    assert vulnerable_lines_from_diff(before, after) == set()


def test_reindentation_is_not_a_vulnerability():
    before = "int f(void) {\n\treturn 1;   \n}"
    after = "int f(void) {\n\treturn 1;\n}"
    assert vulnerable_lines_from_diff(before, after) == set()


def test_removing_a_blank_line_is_not_a_vulnerability():
    before = "a();\n\nb();"
    after = "a();\nb();"
    assert vulnerable_lines_from_diff(before, after) == set()


def test_identical_functions_label_nothing():
    assert vulnerable_lines_from_diff(BEFORE, BEFORE) == set()


def test_line_numbers_are_1_based():
    before = "bad();\nok();"
    after = "good();\nok();"
    assert vulnerable_lines_from_diff(before, after) == {1}


def _write_paired(tmpdir, records):
    p = tmpdir / "test_paired.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return str(p)


def test_loads_paired_records_and_derives_lines(tmp_path):
    path = _write_paired(tmp_path, [
        {"pair_id": "x1", "target": 1, "func": BEFORE, "idx": 7,
         "cwe": ["CWE-787"], "commit_id": "deadbeef"},
        {"pair_id": "x1", "target": 0, "func": AFTER},
    ])
    fns = load_primevul_paired(path)
    assert len(fns) == 1
    fn = fns[0]
    assert fn.vulnerable_lines == {3}
    assert fn.cwe == ["CWE-787"] and fn.commit_id == "deadbeef"
    assert fn.target == 1 and fn.n_lines == 7
    # The patched half is kept so the renderer can show the diff that defines the labels.
    assert fn.func_after == AFTER


def test_functions_fixed_only_by_insertion_are_dropped_not_counted_as_failures(tmp_path):
    path = _write_paired(tmp_path, [
        {"pair_id": "a", "target": 1, "func": "use(p);"},
        {"pair_id": "a", "target": 0, "func": "if (!p) return -1;\nuse(p);"},
    ])
    assert load_primevul_paired(path) == []


def test_inline_func_after_layout_is_supported(tmp_path):
    """Some releases put both halves on the vulnerable record instead of pairing rows."""
    path = _write_paired(tmp_path, [
        {"target": 1, "func": BEFORE, "func_after": AFTER, "idx": 1},
    ])
    fns = load_primevul_paired(path)
    assert len(fns) == 1 and fns[0].vulnerable_lines == {3}


def test_patched_half_is_never_returned(tmp_path):
    path = _write_paired(tmp_path, [
        {"pair_id": "p", "target": 1, "func": BEFORE},
        {"pair_id": "p", "target": 0, "func": AFTER},
    ])
    assert all(f.target == 1 for f in load_primevul_paired(path))


def test_missing_file_names_where_to_get_it():
    with pytest.raises(FileNotFoundError, match="PrimeVul"):
        load_primevul_paired("does/not/exist_paired.jsonl")


def test_line_lengths_ignores_indentation():
    out = line_lengths("    abc\nlonger line here\n")
    assert out[1] == 3
    assert out[2] == len("longer line here")
