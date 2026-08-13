"""Shared helpers: config loading, seeding, device resolution."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import yaml


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str = "config.yaml") -> dict:
    # A bare "config.yaml" resolves relative to the repo root, not the process's cwd, so tests
    # and scripts behave the same whether invoked from the repo root or elsewhere.
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(_REPO_ROOT, path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
