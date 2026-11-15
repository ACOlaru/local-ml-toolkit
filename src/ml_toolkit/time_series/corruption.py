"""Controlled corruption methods for robustness experiments."""

import numpy as np


def add_noise(values, noise_level=0.1, random_state=None):
    """Add Gaussian noise to a time series."""
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative.")

    data = np.asarray(values, dtype=float)

    if data.ndim != 1:
        raise ValueError("Input must be one-dimensional.")

    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, noise_level, size=len(data))

    return data + noise


def add_missing_values(values, missing_rate=0.1, random_state=None):
    """Replace randomly selected observations with NaN."""
    if not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be between 0 and 1.")

    data = np.asarray(values, dtype=float).copy()

    if data.ndim != 1:
        raise ValueError("Input must be one-dimensional.")

    rng = np.random.default_rng(random_state)
    count = int(len(data) * missing_rate)

    if count:
        indices = rng.choice(len(data), size=count, replace=False)
        data[indices] = np.nan

    return data


def add_distribution_shift(values, shift=1.0, start_fraction=0.5):
    """Apply a constant shift to the second part of a series."""
    if not 0 <= start_fraction <= 1:
        raise ValueError("start_fraction must be between 0 and 1.")

    data = np.asarray(values, dtype=float).copy()

    if data.ndim != 1:
        raise ValueError("Input must be one-dimensional.")

    start = int(len(data) * start_fraction)
    data[start:] += shift

    return data


def fill_missing(values, strategy="mean"):
    """Fill missing values using a simple imputation strategy."""
    data = np.asarray(values, dtype=float).copy()

    if data.ndim != 1:
        raise ValueError("Input must be one-dimensional.")

    if np.all(np.isnan(data)):
        raise ValueError("Cannot fill a series containing only missing values.")

    if strategy == "mean":
        fill_value = np.nanmean(data)
        return np.where(np.isnan(data), fill_value, data)

    if strategy == "forward":
        result = data.copy()

        for index in range(len(result)):
            if np.isnan(result[index]) and index > 0:
                result[index] = result[index - 1]

        first_valid = np.flatnonzero(~np.isnan(result))

        if len(first_valid):
            result[:first_valid[0]] = result[first_valid[0]]

        return result

    raise ValueError(f"Unknown filling strategy: {strategy}")