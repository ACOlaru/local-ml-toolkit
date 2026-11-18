"""Synthetic series and controlled anomaly injection with ground-truth labels."""

from __future__ import annotations

import numpy as np

ANOMALY_TYPES = ("spike", "level_shift", "variance_burst", "contextual", "collective")


def make_seasonal_series(n=600, period=24, amplitude=1.0, trend=0.0, noise=0.1, random_state=None):
    """Create a noisy sine wave with an optional linear trend."""
    if n < 1 or period < 2:
        raise ValueError("n must be positive and period must be at least 2.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(random_state)
    t = np.arange(n)
    signal = amplitude * np.sin(2 * np.pi * t / period) + trend * t / n

    return signal + rng.normal(0, noise, size=n)


def _choose_starts(low, high, count, length, rng):
    """Pick `count` non-touching segment starts inside [low, high)."""
    slot = (high - low) // count

    if slot < length + 1:
        raise ValueError(
            f"Cannot place {count} anomalies of length {length} in a region of {high - low} points."
        )

    max_offset = slot - length - 1  # keeps at least one clean point between segments
    return [low + k * slot + int(rng.integers(0, max_offset + 1)) for k in range(count)]


def inject_anomalies(
    values,
    kind="spike",
    count=5,
    magnitude=4.0,
    length=8,
    period=24,
    start_fraction=0.3,
    random_state=None,
):
    """Inject anomalies and return (corrupted_series, labels).

    `magnitude` is expressed in standard deviations of the input series. Anomalies are
    only placed after `start_fraction` of the series so that the beginning stays clean
    for training. `labels` is an integer array with 1 at every anomalous index.

    Kinds:
        spike           single points pushed up or down
        level_shift     a short segment offset from the baseline
        variance_burst  a segment with extra noise
        contextual      a segment copied from the opposite seasonal phase
        collective      a segment frozen at its starting value (stuck sensor)
    """
    if kind not in ANOMALY_TYPES:
        raise ValueError(f"Unknown anomaly kind '{kind}'. Choose from: {', '.join(ANOMALY_TYPES)}")
    if count < 0 or length < 1 or magnitude < 0:
        raise ValueError("count and magnitude must be non-negative and length positive.")
    if not 0 <= start_fraction < 1:
        raise ValueError("start_fraction must be in [0, 1).")

    data = np.asarray(values, dtype=float).copy()

    if data.ndim != 1:
        raise ValueError("Input must be one-dimensional.")

    labels = np.zeros(len(data), dtype=int)

    if count == 0:
        return data, labels

    rng = np.random.default_rng(random_state)
    scale = float(np.std(data)) or 1.0
    segment = 1 if kind == "spike" else length
    low = max(1, int(len(data) * start_fraction))
    high = len(data) - (period // 2 if kind == "contextual" else 0)

    for start in _choose_starts(low, high, count, segment, rng):
        stop = start + segment
        sign = rng.choice([-1.0, 1.0])

        if kind == "spike":
            data[start] += sign * magnitude * scale
        elif kind == "level_shift":
            data[start:stop] += sign * magnitude * scale
        elif kind == "variance_burst":
            data[start:stop] += rng.normal(0, magnitude * scale / 2, size=segment)
        elif kind == "contextual":
            offset = period // 2
            data[start:stop] = data[start + offset : stop + offset]
        else:
            data[start:stop] = data[start - 1]

        labels[start:stop] = 1

    return data, labels
