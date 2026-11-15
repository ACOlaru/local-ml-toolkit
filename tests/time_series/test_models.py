import numpy as np
import pytest

from ml_toolkit.time_series.models import (
    NaiveForecaster,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
)


def test_naive_forecaster():
    model = NaiveForecaster().fit([1, 2, 3])

    assert np.array_equal(model.predict(3), [3, 3, 3])


def test_moving_average_forecaster():
    model = MovingAverageForecaster(window=2).fit([1, 2, 3, 4])

    assert np.array_equal(model.predict(2), [3.5, 3.5])


def test_exponential_smoothing():
    model = ExponentialSmoothingForecaster(alpha=1).fit([1, 2, 3])

    assert np.array_equal(model.predict(2), [3, 3])


def test_invalid_alpha():
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha=0)


def test_invalid_prediction_steps():
    model = NaiveForecaster().fit([1, 2, 3])

    with pytest.raises(ValueError):
        model.predict(0)