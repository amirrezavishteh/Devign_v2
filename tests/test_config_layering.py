"""Tests for `extends:` config inheritance.

These exist because a full-copy machine config silently went stale: `config_a100.yaml` was a
frozen copy of `config.yaml`, so later fixes to dropout, word2vec min_count and early-stopping
patience never reached the GPU, and two full training runs were spent on pre-fix hyperparameters
before anyone noticed. Machine configs must now inherit rather than duplicate.
"""
from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from training.utils import deep_merge, load_config


def _write(dirpath: str, name: str, data: dict) -> str:
    path = os.path.join(dirpath, name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return path


def test_deep_merge_overrides_leaves_and_preserves_siblings():
    base = {"a": {"x": 1, "y": 2}, "b": 3, "c": [1, 2]}
    over = {"a": {"y": 99}, "c": [9]}
    out = deep_merge(base, over)
    assert out["a"] == {"x": 1, "y": 99}   # sibling x preserved, y overridden
    assert out["b"] == 3                    # untouched key preserved
    assert out["c"] == [9]                  # lists replaced wholesale, not merged
    assert base["a"]["y"] == 2              # inputs not mutated


def test_extends_inherits_and_overrides():
    with tempfile.TemporaryDirectory() as d:
        _write(d, "parent.yaml", {"training": {"lr": 0.1, "epochs": 200},
                                  "model": {"dropout": 0.5}})
        child = _write(d, "child.yaml", {"extends": "parent.yaml",
                                         "training": {"epochs": 30}})
        cfg = load_config(child)

    assert cfg["training"]["epochs"] == 30      # child wins
    assert cfg["training"]["lr"] == 0.1         # inherited
    assert cfg["model"]["dropout"] == 0.5       # inherited untouched
    assert "extends" not in cfg                 # directive stripped


def test_extends_resolves_relative_to_the_child_file():
    """The parent path is relative to the config that names it, not to the process cwd."""
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "configs")
        os.makedirs(sub)
        _write(sub, "base.yaml", {"data": {"max_nodes": 500}})
        child = _write(sub, "gpu.yaml", {"extends": "base.yaml",
                                         "data": {"max_nodes": 400}})
        cfg = load_config(child)
    assert cfg["data"]["max_nodes"] == 400


def test_circular_extends_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.yaml")
        b = os.path.join(d, "b.yaml")
        _write(d, "a.yaml", {"extends": "b.yaml"})
        _write(d, "b.yaml", {"extends": "a.yaml"})
        with pytest.raises(ValueError, match="circular"):
            load_config(a)


def test_shipped_a100_config_inherits_the_current_fixes():
    """Guards the exact regression that cost two runs: the A100 config must NOT be a stale copy."""
    cfg = load_config("config_a100.yaml")
    base = load_config("config.yaml")

    # Its own override survives...
    assert cfg["dataset"]["max_nodes_per_batch"] == 24000
    # ...and everything else tracks config.yaml rather than a frozen snapshot.
    assert cfg["embedding"]["word2vec_min_count"] == base["embedding"]["word2vec_min_count"]
    assert cfg["model"]["dropout"] == base["model"]["dropout"]
    assert cfg["training"]["early_stopping_patience"] == base["training"]["early_stopping_patience"]
    assert cfg["training"]["monitor"] == base["training"]["monitor"]
