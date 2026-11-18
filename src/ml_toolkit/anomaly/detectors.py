"""Unsupervised anomaly detectors for univariate time series.

Every detector follows the same small interface:

    detector.fit(train_values)          # learn what "normal" looks like
    scores = detector.score(values)     # one non-negative score per point
    flags = detector.predict(values)    # scores > threshold

Higher scores always mean "more anomalous". Missing values (NaN) are replaced by the
series median before scoring, so detectors never return NaN scores.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MAD_SCALE = 1.4826  # makes the MAD a consistent estimator of the standard deviation
EPS = 1e-12


def prepare_series(values) -> np.ndarray:
    """Validate a series and fill missing values with the median."""
    data = np.asarray(values, dtype=float)

    if data.ndim != 1:
        raise ValueError("Input must be one-dimensional.")
    if data.size == 0:
        raise ValueError("Input series is empty.")
    if np.all(np.isnan(data)):
        raise ValueError("Series contains only missing values.")

    if np.isnan(data).any():
        data = np.where(np.isnan(data), np.nanmedian(data), data)

    return data


def robust_scale(values) -> tuple[float, float]:
    """Return (median, scaled MAD), falling back to the standard deviation."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) * MAD_SCALE)

    if mad < EPS:
        mad = float(np.std(values))

    return median, max(mad, EPS)


def moving_average(values, window: int) -> np.ndarray:
    """Centered moving average with edge padding; output has the input length."""
    if window < 1:
        raise ValueError("window must be at least 1.")

    data = np.asarray(values, dtype=float)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(data, (left, right), mode="edge")

    return np.convolve(padded, np.ones(window) / window, mode="valid")


class BaseDetector:
    """Shared fit/score/predict plumbing."""

    name = "base"
    default_threshold = 3.0

    def __init__(self):
        self._fitted = False

    def fit(self, values):
        self._fit(prepare_series(values))
        self._fitted = True
        return self

    def score(self, values) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.name} detector must be fitted before scoring.")
        return self._score(prepare_series(values))

    def fit_score(self, values) -> np.ndarray:
        return self.fit(values).score(values)

    def predict(self, values, threshold: float | None = None) -> np.ndarray:
        limit = self.default_threshold if threshold is None else threshold
        return self.score(values) > limit

    def _fit(self, data: np.ndarray) -> None:
        raise NotImplementedError

    def _score(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZScoreDetector(BaseDetector):
    """Distance from the training mean in units of standard deviation."""

    name = "zscore"
    default_threshold = 3.0

    def __init__(self, robust: bool = False):
        super().__init__()
        self.robust = robust

    def _fit(self, data):
        if self.robust:
            self.center_, self.spread_ = robust_scale(data)
        else:
            self.center_ = float(np.mean(data))
            self.spread_ = max(float(np.std(data)), EPS)

    def _score(self, data):
        return np.abs(data - self.center_) / self.spread_


class IQRDetector(BaseDetector):
    """Tukey-style detector: distance from the median in interquartile ranges.

    A point 2 IQRs from the median lies exactly on the classic 1.5 * IQR fence.
    """

    name = "iqr"
    default_threshold = 2.0

    def _fit(self, data):
        q1, q3 = np.percentile(data, [25, 75])
        self.median_ = float(np.median(data))
        self.iqr_ = max(float(q3 - q1), EPS)

    def _score(self, data):
        return np.abs(data - self.median_) / self.iqr_


class RollingMADDetector(BaseDetector):
    """Robust z-score against the median and MAD of the preceding window."""

    name = "rolling_mad"
    default_threshold = 3.5

    def __init__(self, window: int = 30):
        super().__init__()
        if window < 3:
            raise ValueError("window must be at least 3.")
        self.window = window

    def _fit(self, data):
        # The detector is local, so there is nothing global to learn.
        self.train_size_ = len(data)

    def _score(self, data):
        n = len(data)
        window = min(self.window, n - 1)

        if window < 3:
            return np.zeros(n)

        windows = sliding_window_view(data, window)
        history = windows[:-1]  # row k holds data[k : k + window], used for index k + window
        medians = np.median(history, axis=1)
        mads = np.median(np.abs(history - medians[:, None]), axis=1) * MAD_SCALE

        # Warm-up region: reuse the statistics of the first full window.
        first_median = np.median(windows[0])
        first_mad = np.median(np.abs(windows[0] - first_median)) * MAD_SCALE
        center = np.concatenate([np.full(window, first_median), medians])
        spread = np.concatenate([np.full(window, first_mad), mads])

        # Flat windows would make the score explode, so enforce a floor.
        floor = max(0.1 * robust_scale(data)[1], EPS)
        spread = np.maximum(spread, floor)

        return np.abs(data - center) / spread


class EWMADetector(BaseDetector):
    """One-step-ahead residuals of an exponentially weighted moving average."""

    name = "ewma"
    default_threshold = 3.5

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha

    def _residuals(self, data):
        residuals = np.zeros(len(data))
        level = data[0]

        for index in range(1, len(data)):
            residuals[index] = data[index] - level
            level = self.alpha * data[index] + (1 - self.alpha) * level

        return residuals

    def _fit(self, data):
        residuals = self._residuals(data)
        self.sigma_ = robust_scale(residuals[1:] if len(residuals) > 1 else residuals)[1]

    def _score(self, data):
        return np.abs(self._residuals(data)) / self.sigma_


class SeasonalResidualDetector(BaseDetector):
    """Remove trend and a repeating seasonal profile, then score the residual.

    The seasonal profile is learned on the training series, which is assumed to start at
    phase 0. Scoring assumes the evaluated series starts at the same phase.
    """

    name = "seasonal"
    default_threshold = 4.0

    def __init__(self, period: int = 24):
        super().__init__()
        if period < 2:
            raise ValueError("period must be at least 2.")
        self.period = period

    def _fit(self, data):
        if len(data) < 2 * self.period:
            raise ValueError("Training series must cover at least two seasonal periods.")

        detrended = data - moving_average(data, self.period)
        profile = np.array([np.median(detrended[phase :: self.period]) for phase in range(self.period)])
        self.profile_ = profile - profile.mean()
        self.sigma_ = robust_scale(self._residual(data))[1]

    def _residual(self, data):
        phases = np.arange(len(data)) % self.period
        return data - moving_average(data, self.period) - self.profile_[phases]

    def _score(self, data):
        return np.abs(self._residual(data)) / self.sigma_


def _c_factor(size: int) -> float:
    """Average path length of an unsuccessful search in a binary search tree."""
    if size <= 1:
        return 0.0
    if size == 2:
        return 1.0

    return 2.0 * (np.log(size - 1) + 0.5772156649) - 2.0 * (size - 1) / size


class IsolationForestDetector(BaseDetector):
    """Small isolation forest over sliding windows of the series.

    Scores lie in (0, 1); values near 0.5 are normal and values towards 1 are anomalous.
    """

    name = "isolation_forest"
    default_threshold = 0.6

    def __init__(self, n_trees: int = 50, sample_size: int = 128, window: int = 8, random_state: int = 0):
        super().__init__()
        if n_trees < 1 or sample_size < 2 or window < 1:
            raise ValueError("n_trees, sample_size and window must be positive.")
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.window = window
        self.random_state = random_state

    def _embed(self, data):
        standardised = (data - self.mean_) / self.std_
        padded = np.concatenate([np.full(self.window - 1, standardised[0]), standardised])
        return sliding_window_view(padded, self.window)

    def _build(self, rows, depth, limit, rng):
        if depth >= limit or len(rows) <= 1:
            return ("leaf", len(rows))

        spans = rows.max(axis=0) - rows.min(axis=0)
        candidates = np.flatnonzero(spans > EPS)

        if len(candidates) == 0:
            return ("leaf", len(rows))

        feature = int(rng.choice(candidates))
        split = rng.uniform(rows[:, feature].min(), rows[:, feature].max())
        mask = rows[:, feature] < split

        return (
            "split",
            feature,
            split,
            self._build(rows[mask], depth + 1, limit, rng),
            self._build(rows[~mask], depth + 1, limit, rng),
        )

    def _fit(self, data):
        self.mean_ = float(np.mean(data))
        self.std_ = max(float(np.std(data)), EPS)
        rows = np.array(self._embed(data))
        rng = np.random.default_rng(self.random_state)

        self.psi_ = min(self.sample_size, len(rows))
        limit = int(np.ceil(np.log2(max(self.psi_, 2))))
        self.trees_ = []

        for _ in range(self.n_trees):
            sample = rows[rng.choice(len(rows), size=self.psi_, replace=False)]
            self.trees_.append(self._build(sample, 0, limit, rng))

    def _walk(self, node, rows, indices, depth, out):
        if node[0] == "leaf":
            out[indices] = depth + _c_factor(node[1])
            return

        _, feature, split, left, right = node
        mask = rows[indices, feature] < split

        if mask.any():
            self._walk(left, rows, indices[mask], depth + 1, out)
        if (~mask).any():
            self._walk(right, rows, indices[~mask], depth + 1, out)

    def _score(self, data):
        rows = np.array(self._embed(data))
        total = np.zeros(len(rows))

        for tree in self.trees_:
            lengths = np.zeros(len(rows))
            self._walk(tree, rows, np.arange(len(rows)), 0, lengths)
            total += lengths

        return 2.0 ** (-(total / self.n_trees) / _c_factor(self.psi_))


DETECTORS = {
    "zscore": ZScoreDetector,
    "iqr": IQRDetector,
    "rolling_mad": RollingMADDetector,
    "ewma": EWMADetector,
    "seasonal": SeasonalResidualDetector,
    "isolation_forest": IsolationForestDetector,
}


def get_detector(name: str, **kwargs) -> BaseDetector:
    """Create a detector by name, e.g. get_detector("seasonal", period=24)."""
    try:
        detector_class = DETECTORS[name]
    except KeyError:
        raise ValueError(f"Unknown detector '{name}'. Choose from: {', '.join(DETECTORS)}") from None

    return detector_class(**kwargs)
