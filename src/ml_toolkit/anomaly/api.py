"""One-call convenience interface for anomaly detection."""

from __future__ import annotations

import numpy as np

from .detectors import get_detector, prepare_series
from .metrics import segments


def detect_anomalies(values, method="rolling_mad", threshold=None, train_fraction=1.0, **detector_kwargs) -> dict:
    """Fit a detector on the start of a series and flag anomalies in the whole series.

    `train_fraction` is the leading share of the series used for fitting; with the
    default of 1.0 the detector is fitted on the series it scores. `threshold` defaults
    to the detector's own recommended value.
    """
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1].")

    series = prepare_series(values)
    detector = get_detector(method, **detector_kwargs)
    limit = detector.default_threshold if threshold is None else float(threshold)

    detector.fit(series[: max(2, int(len(series) * train_fraction))])
    scores = detector.score(series)
    flags = scores > limit

    return {
        "method": method,
        "threshold": limit,
        "scores": scores,
        "flags": flags,
        "indices": np.flatnonzero(flags).tolist(),
        "segments": segments(flags.astype(int)),
    }
