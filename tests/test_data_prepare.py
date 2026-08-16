"""Tests for split logic and the real-dataset loader, added when synthetic-only data was
replaced with the Devign authors' released dataset (see data/hf_devign.py, data/prepare.py)."""
from __future__ import annotations

import pytest

from data.download import RawFunction, load_devign_parquet_dir, load_real
from data.prepare import (codexglue_split, commit_disjoint_split, split_functions,
                          stratified_split3)


def _make_functions(n: int, commits_per: int = 3) -> list[RawFunction]:
    out = []
    for i in range(n):
        out.append(RawFunction(func=f"int f{i}(void){{return {i};}}", target=i % 2,
                               project="p", name=f"f{i}", commit_id=f"c{i // commits_per}"))
    return out


def test_stratified_split3_preserves_class_balance():
    funcs = _make_functions(300)
    train, val, test = stratified_split3(funcs, 0.75, 0.125, 0.125, seed=1)
    assert len(train) + len(val) + len(test) == len(funcs)
    for part in (train, val, test):
        vuln = sum(f.target for f in part)
        # each class was split with the same fractions, so both should land within a few
        # functions of the expected count -- not exactly, since int() truncates per class.
        assert abs(vuln - len(part) / 2) <= 2


def test_stratified_split3_disjoint():
    funcs = _make_functions(200)
    train, val, test = stratified_split3(funcs, 0.75, 0.125, 0.125, seed=7)
    ids = [set(id(f) for f in part) for part in (train, val, test)]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])


def test_commit_disjoint_split_has_no_shared_commits():
    funcs = _make_functions(300, commits_per=4)
    train, val, test = commit_disjoint_split(funcs, 0.75, 0.125, 0.125, seed=3)
    commits = [set(f.commit_id for f in part) for part in (train, val, test)]
    assert not (commits[0] & commits[1])
    assert not (commits[0] & commits[2])
    assert not (commits[1] & commits[2])
    # every function must still show up exactly once across the three parts
    assert len(train) + len(val) + len(test) == len(funcs)


def test_split_functions_paper_split_has_no_test_set():
    funcs = _make_functions(200)
    data_cfg = {"train_split": 0.75, "paper_split": True, "split_by": "random"}
    train, val, test = split_functions(funcs, data_cfg, seed=1)
    assert len(test) == 0
    assert len(train) + len(val) == len(funcs)


def test_split_functions_dispatches_to_commit_mode():
    funcs = _make_functions(200, commits_per=5)
    data_cfg = {"train_split": 0.75, "val_split": 0.125, "test_split": 0.125,
               "split_by": "commit"}
    train, val, test = split_functions(funcs, data_cfg, seed=1)
    commits = [set(f.commit_id for f in part) for part in (train, val, test)]
    assert not (commits[0] & commits[1] & commits[2])
    assert not (commits[0] & commits[1])


def test_load_real_reads_commit_id():
    import json
    import tempfile
    import os

    records = [{"func": "int f(void){return 0;}", "target": 1, "project": "qemu",
               "commit_id": "abc123"}]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        for r in records:
            tf.write(json.dumps(r) + "\n")
        path = tf.name
    try:
        funcs = load_real(path)
        assert len(funcs) == 1
        assert funcs[0].commit_id == "abc123"
        assert funcs[0].project == "qemu"
        assert funcs[0].target == 1
    finally:
        os.unlink(path)


# ----------------------------------------------------------------------------------------------
# CodeXGLUE split preservation
# ----------------------------------------------------------------------------------------------

def _split_labelled(counts: dict[str, int]) -> list[RawFunction]:
    out = []
    for split, n in counts.items():
        for i in range(n):
            out.append(RawFunction(func=f"int {split}{i}(void){{return {i};}}", target=i % 2,
                                   project="qemu", name=f"{split}{i}", split=split))
    return out


def test_codexglue_split_uses_the_released_partition():
    """The release ships a split; `split_by: codexglue` must reproduce it exactly rather than
    re-splitting, or the numbers stop being comparable to published results on this dataset."""
    funcs = _split_labelled({"train": 40, "validation": 8, "test": 6})
    train, val, test = codexglue_split(funcs, seed=1)
    assert (len(train), len(val), len(test)) == (40, 8, 6)
    assert {f.split for f in train} == {"train"}
    assert {f.split for f in val} == {"validation"}
    assert {f.split for f in test} == {"test"}


def test_codexglue_split_refuses_unlabelled_data():
    """Silently re-splitting when the labels are missing would produce numbers that LOOK like
    leaderboard-comparable ones but are not."""
    funcs = _split_labelled({"train": 10}) + _make_functions(5)
    with pytest.raises(ValueError, match="codexglue"):
        codexglue_split(funcs, seed=1)


def test_split_functions_dispatches_to_codexglue():
    funcs = _split_labelled({"train": 20, "validation": 4, "test": 4})
    train, val, test = split_functions(funcs, {"split_by": "codexglue", "train_split": 0.75}, 1)
    assert (len(train), len(val), len(test)) == (20, 4, 4)


def test_unknown_split_by_is_rejected():
    with pytest.raises(ValueError, match="split_by"):
        split_functions(_make_functions(10), {"split_by": "typo", "train_split": 0.75}, 1)


def test_parquet_dir_loader_requires_all_three_splits(tmp_path):
    (tmp_path / "train-00000-of-00001.parquet").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="validation"):
        load_devign_parquet_dir(str(tmp_path))
