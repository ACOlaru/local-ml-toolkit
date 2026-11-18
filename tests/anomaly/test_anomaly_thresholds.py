import numpy as np
import pytest

from ml_toolkit.anomaly.thresholds import (
    best_f1_threshold,
    mad_threshold,
    percentile_threshold,
    pot_threshold,
    select_threshold,
)


def test_percentile_threshold_matches_numpy():
    scores = np.arange(101.0)

    assert percentile_threshold(scores, 90) == np.percentile(scores, 90)


def test_percentile_threshold_validation():
    with pytest.raises(ValueError):
        percentile_threshold([1, 2, 3], q=100)
    with pytest.raises(ValueError):
        percentile_threshold([], q=50)
    with pytest.raises(ValueError):
        percentile_threshold([1.0, np.inf], q=50)


def test_mad_threshold_sits_above_the_median():
    scores = np.abs(np.random.default_rng(0).normal(size=500))

    assert mad_threshold(scores) > np.median(scores)


def test_mad_threshold_handles_constant_scores():
    result = mad_threshold(np.full(50, 2.0))

    assert np.isfinite(result)
    assert result >= 2.0


def test_pot_threshold_on_exponential_tail():
    scores = np.random.default_rng(1).exponential(size=5000)

    threshold = pot_threshold(scores, risk=1e-3)

    assert threshold > np.quantile(scores, 0.95)
    assert 4 < threshold < 12


def test_pot_threshold_falls_back_for_short_inputs():
    scores = np.arange(20.0)

    assert pot_threshold(scores, risk=0.05) == percentile_threshold(scores, 95)


def test_pot_threshold_validation():
    with pytest.raises(ValueError):
        pot_threshold([1.0] * 50, risk=0)
    with pytest.raises(ValueError):
        pot_threshold([1.0] * 50, init_quantile=1.0)


def test_best_f1_threshold_separates_classes():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1])

    threshold = best_f1_threshold(scores, labels)

    assert 0.2 < threshold < 0.8
    assert np.array_equal(scores > threshold, labels.astype(bool))


def test_best_f1_threshold_requires_positive_labels():
    with pytest.raises(ValueError):
        best_f1_threshold([0.1, 0.2], [0, 0])


def test_select_threshold_dispatch():
    scores = np.arange(100.0)

    assert select_threshold(scores, "percentile", q=50) == percentile_threshold(scores, 50)
    assert select_threshold(scores, "mad", k=2) == mad_threshold(scores, 2)
    assert select_threshold(scores, "best_f1", labels=(scores > 80).astype(int)) > 70


def test_select_threshold_errors():
    with pytest.raises(ValueError):
        select_threshold([1.0, 2.0], "best_f1")
    with pytest.raises(ValueError):
        select_threshold([1.0, 2.0], "unknown")
