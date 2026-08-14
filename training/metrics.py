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


def best_threshold(y_true, probs, objective: str = "mcc") -> float:
    """Decision threshold optimising `objective` on this split.

    The model outputs a ranking; 0.5 is an arbitrary cut through it and is rarely optimal,
    especially with an uncalibrated model (an undertrained net emits logits near 0, so every
    probability sits in a razor-thin band around 0.5). Tuned on validation, applied to test.

    Defaults to MCC rather than F1 deliberately. Maximising F1 would re-create the very trap this
    module exists to avoid: on a 45%-positive split, an extreme threshold that labels everything
    vulnerable scores F1 ~60% while being useless. MCC is 0 for any constant predictor, so it
    cannot be won that way, and it balances both classes -- which matters because the paper
    reports accuracy *and* F1, and an F1-optimal threshold can tank accuracy.
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

    def _score(preds) -> float:
        if objective == "mcc":
            from sklearn.metrics import matthews_corrcoef
            return float(matthews_corrcoef(y_true, preds))
        return binary_metrics(y_true, preds)[objective]

    best_thr, best_score = 0.5, -np.inf
    for thr in candidates:
        score = _score((probs >= thr).astype(int))
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr


# Back-compat alias.
def best_f1_threshold(y_true, probs) -> float:
    return best_threshold(y_true, probs, objective="f1")
