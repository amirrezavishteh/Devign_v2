"""The reporting guard: a metric tuned on a split may never be reported on that split.

This is the one class of bug that makes a reproduction look BETTER than it is, which is the
direction that matters. `train_model` tunes the decision threshold on validation; scoring
validation at that threshold then reports a classifier whose operating point already saw those
labels. `scripts/train.py` sidestepped it whenever a test split existed, but `data.paper_split:
true` has no test split, so the biased number was the only one printed.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.metrics import (ThresholdLeakage, majority_baseline, prob_metrics,
                              require_unbiased)


def _scores(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Weakly informative probabilities, so tuning has something to bite on.
    p = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.15, size=n), 0.01, 0.99)
    return y, p


def test_tuned_on_own_split_is_refused():
    y, p = _scores()
    biased = prob_metrics(y, p, threshold=0.62, split="val", fitted_on_split="val")
    with pytest.raises(ThresholdLeakage):
        require_unbiased(biased)


def test_tuned_threshold_on_a_different_split_is_allowed():
    """Applying a val-tuned threshold to test is the correct, unbiased use of it."""
    y, p = _scores()
    ok = prob_metrics(y, p, threshold=0.62, split="test", fitted_on_split="val")
    assert require_unbiased(ok) is ok


def test_untuned_metrics_are_always_allowed():
    y, p = _scores()
    assert require_unbiased(prob_metrics(y, p, 0.5, split="val")) is not None
    # Untagged metrics (older call sites) must not start failing.
    assert require_unbiased(prob_metrics(y, p, 0.5)) is not None


def test_error_message_names_the_split_and_the_fix():
    y, p = _scores()
    biased = prob_metrics(y, p, threshold=0.62, split="val", fitted_on_split="val")
    with pytest.raises(ThresholdLeakage, match="paper_split"):
        require_unbiased(biased)


def test_pr_auc_is_reported_and_beats_chance_on_informative_scores():
    y, p = _scores()
    m = prob_metrics(y, p, 0.5, split="val")
    assert "pr_auc" in m
    # The scores carry real signal, so average precision must exceed the positive rate.
    assert m["pr_auc"] > float(y.mean()) * 100.0


def test_pr_auc_of_a_constant_predictor_is_the_positive_rate():
    y = np.array([1] * 30 + [0] * 70)
    m = prob_metrics(y, np.full(100, 0.5), 0.5)
    assert m["pr_auc"] == pytest.approx(30.0)
    assert m["auc"] == pytest.approx(50.0)


def test_majority_baseline_is_the_class_balance_not_a_result():
    """On a 45%-positive corpus the majority row is what a model has to beat to mean anything."""
    y = np.array([1] * 45 + [0] * 55)
    base = majority_baseline(y, split="test")
    assert base["predicts"] == 0                      # majority class
    assert base["accuracy"] == pytest.approx(55.0)
    assert base["f1"] == pytest.approx(0.0)           # never predicts the positive class
    assert base["auc"] == pytest.approx(50.0)         # a constant has no ranking
    assert base["pr_auc"] == pytest.approx(45.0)
    assert require_unbiased(base) is base


def test_majority_baseline_on_a_positive_majority_split():
    y = np.array([1] * 60 + [0] * 40)
    base = majority_baseline(y)
    assert base["predicts"] == 1
    assert base["accuracy"] == pytest.approx(60.0)
    # Always-positive scores a high F1 on a balanced corpus -- the exact trap that makes F1@0.5
    # useless for model selection here.
    assert base["f1"] == pytest.approx(75.0)
