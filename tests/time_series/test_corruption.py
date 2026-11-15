import numpy as np
import pytest

from ml_toolkit.time_series.corruption import (
    add_noise,
    add_missing_values,
    add_distribution_shift,
    fill_missing,
)


def test_noise_is_reproducible():
    values = [1, 2, 3]

    first = add_noise(values, random_state=42)
    second = add_noise(values, random_state=42)

    assert np.array_equal(first, second)


def test_missing_values():
    result = add_missing_values(
        np.arange(100),
        missing_rate=0.2,
        random_state=42,
    )

    assert np.isnan(result).sum() == 20


def test_distribution_shift():
    result = add_distribution_shift([1, 1, 1, 1], shift=5)

    assert np.array_equal(result, [1, 1, 6, 6])


def test_mean_fill():
    result = fill_missing([1, np.nan, 3])

    assert np.allclose(result, [1, 2, 3])


def test_invalid_missing_rate():
    with pytest.raises(ValueError):
        add_missing_values([1, 2], missing_rate=2)