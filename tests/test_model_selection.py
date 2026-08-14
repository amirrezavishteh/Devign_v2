"""Regression tests for checkpoint selection.

These exist because of a real bug: the trainer selected checkpoints on validation F1 at a fixed
0.5 threshold. On this dataset (43-51% positive) a degenerate "predict almost everything
vulnerable" classifier scores F1 ~60-67%, which a properly trained model balancing precision
against recall never beats at that threshold -- so training locked onto an early, untrained epoch
and reported it. On the real qemu split that meant reporting 53.99% accuracy for a model that
actually reached ~63%.

The fix is to select on AUC, which is exactly 50 for ANY constant predictor and therefore cannot
be won by the degenerate model. These tests assert that property directly.
"""
from __future__ import annotations

import numpy as np

from training.metrics import best_threshold, binary_metrics, prob_metrics


def _labels(n=1000, positive_rate=0.43, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < positive_rate).astype(int)


def test_degenerate_all_positive_scores_high_f1_but_chance_auc():
    """The exact pathology: high F1, useless model. AUC must expose it."""
    y = _labels(positive_rate=0.43)
    probs = np.full(y.shape, 0.99)  # "everything is vulnerable"

    m = prob_metrics(y, probs, threshold=0.5)
    # F1 approaches the all-positive ceiling 2p/(1+p); for p=0.43 that is ~60%.
    assert m["f1"] > 55.0, f"expected the degenerate F1 trap, got {m['f1']:.2f}"
    assert m["recall"] == 100.0
    # ...but it has no discriminative power at all.
    assert m["auc"] == 50.0
    assert abs(m["mcc"]) < 1e-6


def test_auc_prefers_a_real_model_over_the_degenerate_one():
    """monitor='auc' must rank a genuinely discriminative model above the constant one."""
    y = _labels(positive_rate=0.43, seed=1)
    rng = np.random.default_rng(2)
    # Noisy but genuinely informative scores.
    informative = np.clip(0.5 + 0.2 * (2 * y - 1) + rng.normal(0, 0.15, y.shape), 0, 1)
    degenerate = np.full(y.shape, 0.99)

    auc_real = prob_metrics(y, informative)["auc"]
    auc_degenerate = prob_metrics(y, degenerate)["auc"]

    assert auc_real > auc_degenerate, "AUC failed to prefer the informative model"
    assert auc_real > 70.0

    # And the trap it replaces: on F1@0.5 the degenerate model can still look competitive.
    f1_degenerate = prob_metrics(y, degenerate)["f1"]
    assert f1_degenerate > 55.0


def test_tuned_threshold_beats_the_default_half():
    """0.5 is an arbitrary cut through the score ranking; tuning must not do worse."""
    y = _labels(positive_rate=0.30, seed=3)
    rng = np.random.default_rng(4)
    # Shift scores low so 0.5 is a poor cut for a 30%-positive set.
    probs = np.clip(0.25 + 0.25 * (2 * y - 1) + rng.normal(0, 0.1, y.shape), 0, 1)

    thr = best_threshold(y, probs)
    tuned = binary_metrics(y, (probs >= thr).astype(int))
    half = binary_metrics(y, (probs >= 0.5).astype(int))

    assert 0.0 <= thr <= 1.0
    assert tuned["accuracy"] >= half["accuracy"]
    assert tuned["f1"] >= half["f1"]


def test_threshold_tuning_does_not_pick_the_degenerate_cut():
    """The threshold search must not re-create the trap that model selection just fixed.

    On an uninformative model, maximising F1 would drive the threshold to ~0 (label everything
    vulnerable, F1 ~60% on this prior). The MCC objective must refuse that.
    """
    y = _labels(positive_rate=0.43, seed=6)
    rng = np.random.default_rng(7)
    uninformative = rng.random(y.shape)  # no signal at all

    thr_mcc = best_threshold(y, uninformative, objective="mcc")
    preds_mcc = (uninformative >= thr_mcc).astype(int)
    # It must not collapse to predicting a single class for everything.
    assert 0 < preds_mcc.sum() < preds_mcc.size, "MCC threshold produced a constant predictor"

    # Contrast: the F1 objective happily takes the degenerate cut on the same data.
    thr_f1 = best_threshold(y, uninformative, objective="f1")
    preds_f1 = (uninformative >= thr_f1).astype(int)
    assert preds_f1.sum() > preds_mcc.sum(), (
        "expected the F1 objective to skew far more positive than MCC")


def test_prob_metrics_handles_single_class_validation_split():
    """A split with one class has no ROC curve; must not crash or fake an improvement."""
    y = np.ones(50, dtype=int)
    m = prob_metrics(y, np.linspace(0, 1, 50))
    assert m["auc"] == 50.0
    assert m["mcc"] == 0.0


def test_prob_metrics_records_threshold_used():
    y = _labels(seed=5)
    m = prob_metrics(y, np.linspace(0, 1, y.size), threshold=0.37)
    assert m["threshold"] == 0.37


def test_trainer_defaults_to_auc_not_f1():
    """Guards against silently regressing the selection metric back to F1."""
    from training.trainer import TrainConfig

    assert TrainConfig().monitor == "auc"
    assert TrainConfig().tune_threshold is True
