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


def evaluate_devign_imbalanced(cfg, model, device, splits, featurizer, edge_types):
    """Devign's row in Table 3, with the decision threshold calibrated to the 10% prevalence.

    A threshold tuned on the ~46%-positive validation split is wrong for a 10%-positive set: it
    pushes the model to predict almost everything non-vulnerable, driving F1 to ~0 and falsely
    reversing the paper's Q4 finding. So we resample the *validation* split to the same ratio,
    re-derive the threshold there, and apply it to the resampled evaluation split. Only validation
    data informs the threshold, so the reported numbers stay unbiased.

    Returns (per_project_metrics, threshold).
    """
    from data.dataset import DevignDataset, build_samples
    from evaluation.report import per_project_eval
    from scripts.train import _loader, make_collate_from_cfg
    from training.metrics import best_threshold
    from training.trainer import evaluate

    ratio = cfg["data"]["imbalanced_vuln_ratio"]
    seed = cfg["project"]["seed"]
    collate = make_collate_from_cfg(cfg, edge_types)

    def _score(functions):
        samples = build_samples(functions, featurizer, edge_types, cfg["data"]["max_nodes"])
        if not samples:
            return None, None, None
        ds = DevignDataset(samples, len(featurizer.type_vocab))
        _, probs, labels, projects = evaluate(model, _loader(ds, cfg, collate, shuffle=False),
                                             device)
        return probs, labels, projects

    # Calibrate on an imbalanced validation subsample (never on the reported split).
    cal_probs, cal_labels, _ = _score(make_imbalanced(splits["val"], ratio, seed))
    threshold = 0.5 if cal_probs is None else best_threshold(cal_labels, cal_probs, "mcc")

    eval_split = splits.get("test") or splits["val"]
    probs, labels, projects = _score(make_imbalanced(eval_split, ratio, seed))
    if probs is None:
        return None, threshold
    return per_project_eval(probs, labels, projects, threshold=threshold), threshold


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
