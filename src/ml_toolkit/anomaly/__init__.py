"""Time-series anomaly detection, injection and benchmarking."""

from .api import detect_anomalies
from .benchmark import AnomalyBenchmark, summarize_results
from .detectors import (
    DETECTORS,
    EWMADetector,
    IQRDetector,
    IsolationForestDetector,
    RollingMADDetector,
    SeasonalResidualDetector,
    ZScoreDetector,
    get_detector,
)
from .injection import ANOMALY_TYPES, inject_anomalies, make_seasonal_series
from .metrics import evaluate_detection
from .report import format_anomaly_report
from .thresholds import select_threshold

__all__ = [
    "ANOMALY_TYPES",
    "DETECTORS",
    "AnomalyBenchmark",
    "EWMADetector",
    "IQRDetector",
    "IsolationForestDetector",
    "RollingMADDetector",
    "SeasonalResidualDetector",
    "ZScoreDetector",
    "detect_anomalies",
    "evaluate_detection",
    "format_anomaly_report",
    "get_detector",
    "inject_anomalies",
    "make_seasonal_series",
    "select_threshold",
    "summarize_results",
]
