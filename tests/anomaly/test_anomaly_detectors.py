import numpy as np
import pytest

from ml_toolkit.anomaly.detectors import (
    EWMADetector,
    IQRDetector,
    IsolationForestDetector,
    RollingMADDetector,
    SeasonalResidualDetector,
    ZScoreDetector,
    get_detector,
    moving_average,
    prepare_series,
)
from ml_toolkit.anomaly.injection import make_seasonal_series


def noisy_series(n=300, seed=0):
    return np.random.default_rng(seed).normal(0, 1, size=n)


def test_zscore_flags_spike():
    data = noisy_series()
    detector = ZScoreDetector().fit(data)
    test = data.copy()
    test[100] += 12

    scores = detector.score(test)

    assert scores[100] == scores.max()
    assert detector.predict(test)[100]


def test_robust_zscore_ignores_contaminated_training_data():
    data = noisy_series()
    data[10] = 500
    plain = ZScoreDetector().fit(data)
    robust = ZScoreDetector(robust=True).fit(data)

    assert robust.spread_ < plain.spread_


def test_iqr_flags_spike():
    data = noisy_series()
    detector = IQRDetector().fit(data)
    test = data.copy()
    test[50] = 15

    assert detector.predict(test)[50]


def test_rolling_mad_flags_spike_only():
    test = noisy_series(seed=3)
    test[200] = 20

    flags = RollingMADDetector(window=30).fit(test).predict(test)

    assert flags[200]
    assert flags.sum() < 10


def test_rolling_mad_handles_tiny_series():
    scores = RollingMADDetector(window=30).fit([1.0, 2.0]).score([1.0, 2.0])

    assert np.all(scores == 0)


def test_ewma_flags_spike():
    test = noisy_series(seed=5)
    test[150] += 15

    flags = EWMADetector(alpha=0.3).fit(test[:100]).predict(test)

    assert flags[150]


def test_seasonal_detector_flags_spike_in_seasonal_series():
    series = make_seasonal_series(240, period=24, noise=0.05, random_state=1)
    detector = SeasonalResidualDetector(period=24).fit(series[:120])
    test = series.copy()
    test[200] += 3

    flags = detector.predict(test)

    assert flags[200]


def test_seasonal_detector_needs_two_periods():
    with pytest.raises(ValueError):
        SeasonalResidualDetector(period=24).fit(np.arange(30.0))


def test_isolation_forest_scores_spike_windows_higher():
    series = make_seasonal_series(200, period=24, noise=0.05, random_state=2)
    detector = IsolationForestDetector(window=6, random_state=0).fit(series)
    test = series.copy()
    test[100] += 6

    scores = detector.score(test)

    assert np.all((scores > 0) & (scores < 1))
    assert scores[100:106].mean() > scores.mean()


def test_isolation_forest_is_reproducible():
    series = make_seasonal_series(150, period=24, noise=0.1, random_state=4)
    first = IsolationForestDetector(random_state=7).fit_score(series)
    second = IsolationForestDetector(random_state=7).fit_score(series)

    assert np.array_equal(first, second)


def test_scoring_before_fit_raises():
    with pytest.raises(RuntimeError):
        ZScoreDetector().score([1.0, 2.0, 3.0])


def test_missing_values_are_filled():
    data = noisy_series(100)
    data[[5, 50]] = np.nan

    scores = ZScoreDetector().fit(data).score(data)

    assert np.all(np.isfinite(scores))


def test_prepare_series_validation():
    with pytest.raises(ValueError):
        prepare_series(np.ones((3, 3)))
    with pytest.raises(ValueError):
        prepare_series([])
    with pytest.raises(ValueError):
        prepare_series([np.nan, np.nan])


def test_constructor_validation():
    with pytest.raises(ValueError):
        RollingMADDetector(window=2)
    with pytest.raises(ValueError):
        EWMADetector(alpha=0)
    with pytest.raises(ValueError):
        SeasonalResidualDetector(period=1)
    with pytest.raises(ValueError):
        IsolationForestDetector(n_trees=0)


def test_moving_average_keeps_length_and_constants():
    result = moving_average(np.full(20, 3.0), 5)

    assert len(result) == 20
    assert np.allclose(result, 3.0)


def test_get_detector_by_name():
    assert isinstance(get_detector("seasonal", period=12), SeasonalResidualDetector)

    with pytest.raises(ValueError):
        get_detector("does_not_exist")
