"""Benchmark anomaly detectors across anomaly types and noise levels."""

from __future__ import annotations

import time

import numpy as np

from .detectors import (
    EWMADetector,
    IQRDetector,
    IsolationForestDetector,
    RollingMADDetector,
    SeasonalResidualDetector,
    ZScoreDetector,
)
from .injection import ANOMALY_TYPES, inject_anomalies, make_seasonal_series
from .metrics import evaluate_detection
from .thresholds import select_threshold

METRIC_KEYS = (
    "f1",
    "precision",
    "recall",
    "point_adjusted_f1",
    "range_f1",
    "roc_auc",
    "average_precision",
    "detection_delay",
    "false_alarm_rate",
    "runtime_seconds",
)


def default_detectors(period: int = 24) -> dict:
    """Factories for the standard detector line-up, keyed by name."""
    return {
        "zscore": lambda: ZScoreDetector(),
        "iqr": lambda: IQRDetector(),
        "rolling_mad": lambda: RollingMADDetector(window=max(3, period)),
        "ewma": lambda: EWMADetector(alpha=0.3),
        "seasonal": lambda: SeasonalResidualDetector(period=period),
        "isolation_forest": lambda: IsolationForestDetector(window=max(2, period // 3)),
    }


class AnomalyBenchmark:
    """Run every detector on synthetic series with known anomalies.

    Each detector is fitted on a clean leading part of the series, thresholded using
    scores from that same clean part (so no labels are used), and evaluated on the rest.
    With `threshold_method="best_f1"` the threshold is instead tuned on the evaluation
    labels, which gives an upper bound on what a perfect threshold could achieve.
    """

    def __init__(
        self,
        detectors=None,
        anomaly_types=ANOMALY_TYPES,
        noise_levels=(0.1, 0.3),
        n=600,
        period=24,
        train_fraction=0.3,
        threshold_method="mad",
        threshold_kwargs=None,
        count=6,
        magnitude=4.0,
        length=8,
        random_state=0,
    ):
        if not 0.1 <= train_fraction <= 0.8:
            raise ValueError("train_fraction must be between 0.1 and 0.8.")
        if n < 4 * period:
            raise ValueError("n must cover at least four seasonal periods.")

        self.detectors = detectors if detectors is not None else default_detectors(period)
        self.anomaly_types = tuple(anomaly_types)
        self.noise_levels = tuple(noise_levels)
        self.n = n
        self.period = period
        self.train_fraction = train_fraction
        self.threshold_method = threshold_method
        self.threshold_kwargs = dict(threshold_kwargs or {})
        self.count = count
        self.magnitude = magnitude
        self.length = length
        self.random_state = random_state

        unknown = [kind for kind in self.anomaly_types if kind not in ANOMALY_TYPES]
        if unknown:
            raise ValueError(f"Unknown anomaly types: {', '.join(unknown)}")

    def _make_case(self, kind, noise, seed):
        base = make_seasonal_series(self.n, self.period, noise=noise, random_state=seed)

        return inject_anomalies(
            base,
            kind=kind,
            count=self.count,
            magnitude=self.magnitude,
            length=self.length,
            period=self.period,
            start_fraction=self.train_fraction,
            random_state=seed + 1,
        )

    def _run_detector(self, name, factory, series, labels, train_size):
        detector = factory()
        start = time.perf_counter()
        detector.fit(series[:train_size])
        scores = detector.score(series)
        runtime = time.perf_counter() - start

        if self.threshold_method == "best_f1":
            threshold = select_threshold(scores[train_size:], "best_f1", labels=labels[train_size:])
        else:
            threshold = select_threshold(scores[:train_size], self.threshold_method, **self.threshold_kwargs)

        flags = scores > threshold
        metrics = evaluate_detection(labels[train_size:], flags[train_size:], scores[train_size:])

        return {
            "detector": name,
            "threshold": float(threshold),
            "runtime_seconds": runtime,
            **metrics,
        }

    def run(self) -> list[dict]:
        """Return one result row per (anomaly type, noise level, detector)."""
        train_size = int(self.n * self.train_fraction)
        results = []

        for kind_index, kind in enumerate(self.anomaly_types):
            for noise_index, noise in enumerate(self.noise_levels):
                seed = self.random_state + 100 * kind_index + 10 * noise_index
                series, labels = self._make_case(kind, noise, seed)

                for name, factory in self.detectors.items():
                    row = self._run_detector(name, factory, series, labels, train_size)
                    row.update({"anomaly_type": kind, "noise": noise})
                    results.append(row)

        return results


def summarize_results(results, by="detector") -> list[dict]:
    """Average every metric over the rows that share the same `by` value.

    NaN entries (for example the delay of a detector that found nothing) are ignored.
    Rows are returned sorted by mean F1, best first.
    """
    if not results:
        raise ValueError("No benchmark results supplied.")

    groups = {}
    for row in results:
        groups.setdefault(row[by], []).append(row)

    summary = []
    for key, rows in groups.items():
        entry = {by: key, "cases": len(rows)}

        for metric in METRIC_KEYS:
            values = np.array([row[metric] for row in rows if metric in row], dtype=float)
            values = values[~np.isnan(values)]
            entry[metric] = float(values.mean()) if len(values) else float("nan")

        summary.append(entry)

    return sorted(summary, key=lambda item: item["f1"], reverse=True)


def best_by_group(results, by="anomaly_type", metric="f1") -> dict:
    """Return the best-scoring detector name for each group."""
    best = {}

    for group in sorted({row[by] for row in results}):
        rows = [row for row in results if row[by] == group]
        summary = summarize_results([{**row, "detector": row["detector"]} for row in rows])
        best[group] = max(summary, key=lambda item: np.nan_to_num(item[metric], nan=-1.0))["detector"]

    return best
