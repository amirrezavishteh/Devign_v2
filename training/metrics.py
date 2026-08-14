"""Accuracy and F1 (the two metrics the paper's tables report), plus the threshold-free metrics
used for model selection.

Why AUC matters here: this dataset is 43-51% positive, so a degenerate "predict almost everything
vulnerable" classifier scores F1 ~60-67% at a 0.5 threshold while being useless -- and a properly
trained model that balances precision against recall cannot beat it at that threshold. Selecting
checkpoints on F1@0.5 therefore locks onto an early, barely-trained epoch. AUC is immune to this:
a constant predictor scores exactly 50 no matter what constant it emits.
"""
from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = max(1, len(y_true))
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": acc * 100.0,
        "f1": f1 * 100.0,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def prob_metrics(y_true, probs, threshold: float = 0.5) -> dict[str, float]:
    """`binary_metrics` at `threshold`, plus threshold-free AUC and MCC.

    AUC and MCC are what model selection should key on; accuracy/F1 are what the paper reports.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= threshold).astype(int)
    out = binary_metrics(y_true, preds)

    # A validation split with a single class present has no defined ROC curve; 50.0 is the
    # no-skill value and keeps early stopping from treating it as an improvement.
    if len(np.unique(y_true)) < 2:
        out["auc"] = 50.0
        out["mcc"] = 0.0
    else:
        from sklearn.metrics import matthews_corrcoef, roc_auc_score
        out["auc"] = float(roc_auc_score(y_true, probs)) * 100.0
        out["mcc"] = float(matthews_corrcoef(y_true, preds)) * 100.0

    out["threshold"] = float(threshold)
    return out


def best_threshold(y_true, probs, objective: str = "f1_guarded") -> float:
    """Decision threshold optimising `objective` on this split.

    The model outputs a ranking; 0.5 is an arbitrary cut through it and is rarely optimal,
    especially with an uncalibrated model (an undertrained net emits logits near 0, so every
    probability sits in a razor-thin band around 0.5). Tuned on validation, applied to test.

    Objectives:

    ``f1_guarded`` (default)
        Maximise F1 **subject to accuracy >= accuracy at 0.5**. Two properties motivate it:

        1. It provably dominates the 0.5 default on *both* metrics the paper reports, because
           0.5 is always in the feasible set -- so F1 >= F1@0.5 and accuracy >= accuracy@0.5.
        2. It cannot pick the degenerate all-positive cut, because labelling everything
           vulnerable craters accuracy (40.9% on qemu vs ~62% at 0.5) and is excluded.

        Plain ``f1`` fails (2): on a 45%-positive split, "everything is vulnerable" scores
        F1 ~60% while being useless. Plain ``mcc`` fails a different way, observed on real data:
        it chose threshold 0.996, buying 4 points of accuracy for 14 points of F1 (65.69/37.96).

    ``mcc`` / ``f1`` / ``accuracy``
        Unconstrained maximisation of that single metric. Kept for analysis; see the caveats above.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    if len(np.unique(y_true)) < 2 or probs.size == 0:
        return 0.5

    # Candidates: midpoints between consecutive distinct predicted probabilities, capped so a
    # large validation split doesn't turn this into an O(n^2) sweep.
    uniq = np.unique(probs)
    if uniq.size > 512:
        uniq = np.quantile(uniq, np.linspace(0.0, 1.0, 512))
    candidates = np.unique(np.concatenate([[0.5], (uniq[:-1] + uniq[1:]) / 2.0]))

    def _mcc(preds) -> float:
        from sklearn.metrics import matthews_corrcoef
        return float(matthews_corrcoef(y_true, preds))

    if objective == "f1_guarded":
        floor = binary_metrics(y_true, (probs >= 0.5).astype(int))["accuracy"]
        best_thr, best_f1 = 0.5, -np.inf
        for thr in candidates:
            m = binary_metrics(y_true, (probs >= thr).astype(int))
            # Strictly-worse-accuracy thresholds are infeasible; ties are allowed so the search
            # can still trade a flat accuracy region for better F1.
            if m["accuracy"] + 1e-9 < floor:
                continue
            if m["f1"] > best_f1:
                best_f1, best_thr = m["f1"], float(thr)
        return best_thr

    def _score(preds) -> float:
        return _mcc(preds) if objective == "mcc" else binary_metrics(y_true, preds)[objective]

    best_thr, best_score = 0.5, -np.inf
    for thr in candidates:
        score = _score((probs >= thr).astype(int))
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr


# Back-compat alias.
def best_f1_threshold(y_true, probs) -> float:
    return best_threshold(y_true, probs, objective="f1")
