"""Shared helpers: config loading, seeding, device resolution."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import yaml


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_config_path(path: str, relative_to: str | None = None) -> str:
    """Resolve a config path against its referrer's directory, then the repo root, then cwd."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidates = []
    if relative_to:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(relative_to)), path))
    candidates.append(os.path.join(_REPO_ROOT, path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base`, returning a new dict.

    Mappings merge key-by-key; every other type (including lists) is replaced wholesale, so an
    override can redefine e.g. `data.projects` without inheriting stale entries.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str = "config.yaml", _seen: tuple = ()) -> dict:
    """Load a config, applying `extends:` inheritance.

    A machine-specific config should contain ONLY the keys that differ, e.g.

        extends: config.yaml
        dataset:
          max_nodes_per_batch: 24000

    and is deep-merged over its parent. This exists because full-copy configs silently go stale:
    an earlier `config_a100.yaml` was a frozen copy of `config.yaml`, so later fixes to dropout,
    word2vec min_count and early-stopping patience never reached the GPU and two full training
    runs were wasted on pre-fix hyperparameters.
    """
    # A bare "config.yaml" resolves relative to the repo root, not the process's cwd, so tests
    # and scripts behave the same whether invoked from the repo root or elsewhere.
    path = _resolve_config_path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent_ref = cfg.pop("extends", None)
    if not parent_ref:
        return cfg

    real = os.path.abspath(path)
    if real in _seen:
        chain = " -> ".join(_seen + (real,))
        raise ValueError(f"circular config `extends` chain: {chain}")

    parent_path = _resolve_config_path(str(parent_ref), relative_to=path)
    parent = load_config(parent_path, _seen=_seen + (real,))
    return deep_merge(parent, cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
