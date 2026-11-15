import pytest

from ml_toolkit.time_series.metrics import mae, rmse, mape


def test_mae():
    assert mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)


def test_rmse():
    assert rmse([1, 2, 3], [2, 2, 2]) == pytest.approx((2 / 3) ** 0.5)


def test_mape():
    assert mape([100, 200], [110, 180]) == pytest.approx(10)


def test_different_lengths():
    with pytest.raises(ValueError):
        mae([1, 2], [1])


def test_empty_arrays():
    with pytest.raises(ValueError):
        rmse([], [])