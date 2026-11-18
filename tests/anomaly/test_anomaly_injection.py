import numpy as np
import pytest

from ml_toolkit.anomaly.injection import ANOMALY_TYPES, inject_anomalies, make_seasonal_series
from ml_toolkit.anomaly.metrics import segments


def base_series():
    return make_seasonal_series(600, period=24, noise=0.1, random_state=0)


def test_make_seasonal_series_is_reproducible():
    first = make_seasonal_series(100, random_state=3)
    second = make_seasonal_series(100, random_state=3)

    assert len(first) == 100
    assert np.array_equal(first, second)


def test_make_seasonal_series_validation():
    with pytest.raises(ValueError):
        make_seasonal_series(0)
    with pytest.raises(ValueError):
        make_seasonal_series(100, noise=-1)


@pytest.mark.parametrize("kind", ANOMALY_TYPES)
def test_labels_mark_changed_points_after_clean_start(kind):
    base = base_series()
    corrupted, labels = inject_anomalies(base, kind=kind, count=5, random_state=1)

    assert labels[: int(len(base) * 0.3)].sum() == 0
    assert labels.sum() > 0
    assert np.array_equal(corrupted[labels == 0], base[labels == 0])
    assert np.any(corrupted[labels == 1] != base[labels == 1])


def test_spike_labels_one_point_per_anomaly():
    _, labels = inject_anomalies(base_series(), kind="spike", count=5, random_state=2)

    assert labels.sum() == 5
    assert len(segments(labels)) == 5


def test_segment_kinds_label_full_length():
    _, labels = inject_anomalies(base_series(), kind="level_shift", count=4, length=10, random_state=2)

    assert labels.sum() == 40
    assert all(stop - start == 10 for start, stop in segments(labels))


def test_collective_anomaly_is_flat():
    corrupted, labels = inject_anomalies(base_series(), kind="collective", count=3, length=12, random_state=4)

    for start, stop in segments(labels):
        assert np.ptp(corrupted[start:stop]) == 0


def test_injection_does_not_modify_input_and_is_reproducible():
    base = base_series()
    original = base.copy()

    first = inject_anomalies(base, kind="spike", random_state=9)
    second = inject_anomalies(base, kind="spike", random_state=9)

    assert np.array_equal(base, original)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_zero_count_returns_unchanged_copy():
    base = base_series()
    corrupted, labels = inject_anomalies(base, count=0)

    assert labels.sum() == 0
    assert np.array_equal(corrupted, base)
    assert corrupted is not base


def test_invalid_arguments_raise():
    base = base_series()

    with pytest.raises(ValueError):
        inject_anomalies(base, kind="nonsense")
    with pytest.raises(ValueError):
        inject_anomalies(base, start_fraction=1.0)
    with pytest.raises(ValueError):
        inject_anomalies(base, length=0)
    with pytest.raises(ValueError):
        inject_anomalies(base, kind="level_shift", count=100, length=20)
    with pytest.raises(ValueError):
        inject_anomalies(np.ones((4, 4)))
