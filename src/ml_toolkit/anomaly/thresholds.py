"""Choose a score threshold for turning anomaly scores into alerts."""

from __future__ import annotations

import numpy as np

from .detectors import EPS, MAD_SCALE

THRESHOLD_METHODS = ("percentile", "mad", "pot", "best_f1")


def _as_scores(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite.")

    return values


def percentile_threshold(scores, q: float = 99.0) -> float:
    """Flag everything above the q-th percentile of the scores."""
    if not 0 < q < 100:
        raise ValueError("q must be between 0 and 100.")

    return float(np.percentile(_as_scores(scores), q))


def mad_threshold(scores, k: float = 4.0) -> float:
    """Median plus k robust standard deviations of the scores."""
    values = _as_scores(scores)
    median = float(np.median(values))
    spread = float(np.median(np.abs(values - median)) * MAD_SCALE)

    if spread < EPS:
        spread = float(np.std(values))

    return median + k * max(spread, EPS)


def pot_threshold(scores, risk: float = 1e-3, init_quantile: float = 0.95) -> float:
    """Peaks-over-threshold: fit a generalised Pareto tail and extrapolate.

    `risk` is the target probability that a normal point exceeds the threshold. The tail
    parameters come from the method of moments. With too few exceedances to fit a tail,
    this falls back to a plain percentile.
    """
    if not 0 < risk < 1:
        raise ValueError("risk must be between 0 and 1.")
    if not 0 < init_quantile < 1:
        raise ValueError("init_quantile must be between 0 and 1.")

    values = _as_scores(scores)
    initial = float(np.quantile(values, init_quantile))
    excesses = values[values > initial] - initial

    if len(excesses) < 10:
        return percentile_threshold(values, 100 * (1 - risk))

    mean = float(excesses.mean())
    variance = float(excesses.var())

    if variance < EPS:
        return initial + mean

    ratio = mean * mean / variance
    shape = 0.5 * (1 - ratio)
    scale = 0.5 * mean * (1 + ratio)
    tail_risk = risk * len(values) / len(excesses)

    if abs(shape) < 1e-6:
        threshold = initial - scale * np.log(tail_risk)
    else:
        threshold = initial + scale / shape * (tail_risk ** (-shape) - 1)

    return float(max(threshold, initial))


def best_f1_threshold(scores, labels) -> float:
    """Oracle threshold that maximises point-wise F1 on labelled data."""
    values = _as_scores(scores)
    truth = np.asarray(labels).astype(int)

    if truth.shape != values.shape:
        raise ValueError("scores and labels must have the same length.")

    positives = int(truth.sum())

    if positives == 0:
        raise ValueError("labels must contain at least one anomaly.")

    order = np.argsort(-values, kind="stable")
    ranked_scores = values[order]
    hits = np.cumsum(truth[order])
    flagged = np.arange(1, len(values) + 1)

    precision = hits / flagged
    recall = hits / positives
    f1 = 2 * precision * recall / np.maximum(precision + recall, EPS)

    # Only cut between distinct score values, otherwise ties would be split.
    boundary = np.append(ranked_scores[:-1] != ranked_scores[1:], True)
    f1 = np.where(boundary, f1, -1.0)
    best = int(np.argmax(f1))

    if best + 1 < len(values):
        return float((ranked_scores[best] + ranked_scores[best + 1]) / 2)

    return float(ranked_scores[best] - 1e-9 * max(1.0, abs(ranked_scores[best])))


def select_threshold(scores, method: str = "mad", labels=None, **kwargs) -> float:
    """Dispatch to one of THRESHOLD_METHODS; extra kwargs go to the chosen method."""
    if method == "percentile":
        return percentile_threshold(scores, **kwargs)
    if method == "mad":
        return mad_threshold(scores, **kwargs)
    if method == "pot":
        return pot_threshold(scores, **kwargs)
    if method == "best_f1":
        if labels is None:
            raise ValueError("best_f1 requires labels.")
        return best_f1_threshold(scores, labels)

    raise ValueError(f"Unknown threshold method '{method}'. Choose from: {', '.join(THRESHOLD_METHODS)}")
