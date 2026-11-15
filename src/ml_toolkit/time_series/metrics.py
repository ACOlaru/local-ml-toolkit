"""Evaluation metrics for time-series forecasting."""

import numpy as np


def _validate_arrays(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(actual) != len(predicted):
        raise ValueError("Inputs must have the same length.")

    if len(actual) == 0:
        raise ValueError("Inputs cannot be empty.")

    if not np.all(np.isfinite(actual)):
        raise ValueError("Actual values must be finite.")

    if not np.all(np.isfinite(predicted)):
        raise ValueError("Predicted values must be finite.")

    return actual, predicted


def mae(actual, predicted):
    """Calculate mean absolute error."""
    actual, predicted = _validate_arrays(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    """Calculate root mean squared error."""
    actual, predicted = _validate_arrays(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual, predicted):
    """Calculate mean absolute percentage error.

    Zero actual values are excluded to avoid division by zero.
    """
    actual, predicted = _validate_arrays(actual, predicted)

    nonzero = actual != 0

    if not np.any(nonzero):
        raise ValueError("MAPE cannot be calculated when all actual values are zero.")

    return float(
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
            )
        ) * 100
    )


def evaluate_forecast(actual, predicted):
    """Return several forecasting metrics."""
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
    }