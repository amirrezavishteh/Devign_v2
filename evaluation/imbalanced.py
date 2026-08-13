"""Q4 / Table 3: imbalanced evaluation vs static analyzers.

Builds an imbalanced test set with `imbalanced_vuln_ratio` (10%) vulnerable functions by
subsampling the validation split, then compares Devign (Composite) against Cppcheck / Flawfinder
and the neural baselines, per-project and combined.
"""
from __future__ import annotations

import random

import numpy as np

from evaluation.static_analyzers import run_static_analyzer
from training.metrics import binary_metrics


def make_imbalanced(functions, vuln_ratio: float, seed: int):
    rng = random.Random(seed)
    vuln = [f for f in functions if f.target == 1]
    safe = [f for f in functions if f.target == 0]
    rng.shuffle(vuln)
    rng.shuffle(safe)
    # keep all safe, downsample vuln so vuln/(vuln+safe) == vuln_ratio
    n_safe = len(safe)
    n_vuln = int(round(n_safe * vuln_ratio / (1 - vuln_ratio)))
    n_vuln = min(n_vuln, len(vuln))
    subset = safe + vuln[:n_vuln]
    rng.shuffle(subset)
    return subset


def per_project_metrics(y_true, y_pred, projects):
    out = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    projects = np.array(projects)
    for proj in sorted(set(projects)):
        m = projects == proj
        out[proj] = binary_metrics(y_true[m], y_pred[m])
    out["Combined"] = binary_metrics(y_true, y_pred)
    return out


def evaluate_static_analyzers(functions, tool_paths: dict | None = None):
    """tool_paths: {'cppcheck': path_or_None, 'flawfinder': path_or_None} from
    config.yaml's evaluation.static_analyzer_paths -- explicit path when the binary isn't on
    PATH, or None to look it up on PATH."""
    tool_paths = tool_paths or {}
    results = {}
    for name in ("cppcheck", "flawfinder"):
        y_true, y_pred, projects, mode = run_static_analyzer(name, functions, tool_paths.get(name))
        label = name if mode == "tool" else f"{name} (heuristic-fallback)"
        results[label] = {"metrics": per_project_metrics(y_true, y_pred, projects), "mode": mode}
    return results
