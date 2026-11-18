"""Evaluation metrics for anomaly detection.

Point metrics treat every index independently. Event metrics (point-adjusted and
range-based) credit a detector for catching an anomalous segment even when it flags
only part of it, which matches how alerts are used in practice.
"""

from __future__ import annotations

import numpy as np


def _as_binary_pair(y_true, y_pred):
    truth = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)

    if truth.ndim != 1 or pred.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    return truth, pred


def confusion_counts(y_true, y_pred) -> dict:
    """Return true/false positives and negatives."""
    truth, pred = _as_binary_pair(y_true, y_pred)

    return {
        "tp": int(np.sum((truth == 1) & (pred == 1))),
        "fp": int(np.sum((truth == 0) & (pred == 1))),
        "fn": int(np.sum((truth == 1) & (pred == 0))),
        "tn": int(np.sum((truth == 0) & (pred == 0))),
    }


def precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]:
    """Point-wise precision, recall and F1 (0.0 when undefined)."""
    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1


def false_alarm_rate(y_true, y_pred) -> float:
    """Share of normal points that were flagged."""
    counts = confusion_counts(y_true, y_pred)
    normal = counts["fp"] + counts["tn"]

    return counts["fp"] / normal if normal else 0.0


def segments(labels) -> list[tuple[int, int]]:
    """Return (start, stop) pairs for every run of ones; stop is exclusive."""
    flags = np.asarray(labels).astype(int)
    edges = np.diff(np.concatenate([[0], flags, [0]]))

    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def point_adjust(y_true, y_pred) -> np.ndarray:
    """Mark a whole true segment as detected if any point inside it was flagged."""
    truth, pred = _as_binary_pair(y_true, y_pred)
    adjusted = pred.copy()

    for start, stop in segments(truth):
        if pred[start:stop].any():
            adjusted[start:stop] = 1

    return adjusted


def point_adjusted_f1(y_true, y_pred) -> float:
    """F1 after point adjustment."""
    return precision_recall_f1(y_true, point_adjust(y_true, y_pred))[2]


def range_scores(y_true, y_pred) -> tuple[float, float, float]:
    """Event-level precision, recall and F1.

    Recall is the share of true segments overlapped by any prediction; precision is the
    share of predicted segments that overlap a true segment.
    """
    truth, pred = _as_binary_pair(y_true, y_pred)
    true_segments = segments(truth)
    pred_segments = segments(pred)

    if not true_segments or not pred_segments:
        return 0.0, 0.0, 0.0

    caught = sum(1 for start, stop in true_segments if pred[start:stop].any())
    useful = sum(1 for start, stop in pred_segments if truth[start:stop].any())

    recall = caught / len(true_segments)
    precision = useful / len(pred_segments)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1


def detection_delay(y_true, y_pred) -> float:
    """Mean number of points between segment start and first detection (NaN if none)."""
    truth, pred = _as_binary_pair(y_true, y_pred)
    delays = []

    for start, stop in segments(truth):
        hits = np.flatnonzero(pred[start:stop])
        if len(hits):
            delays.append(int(hits[0]))

    return float(np.mean(delays)) if delays else float("nan")


def _average_ranks(scores) -> np.ndarray:
    """Ranks starting at 1, with ties given the average of their ranks."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    middle = upper - (counts - 1) / 2.0

    return middle[inverse]


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve via the rank-sum formulation (NaN if one class missing)."""
    truth = np.asarray(y_true).astype(int)
    values = np.asarray(scores, dtype=float)

    if truth.shape != values.shape:
        raise ValueError("y_true and scores must have the same length.")

    positives = int(truth.sum())
    negatives = len(truth) - positives

    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = _average_ranks(values)
    return float((ranks[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y_true, scores) -> float:
    """Area under the precision-recall curve (average precision)."""
    truth = np.asarray(y_true).astype(int)
    values = np.asarray(scores, dtype=float)

    if truth.shape != values.shape:
        raise ValueError("y_true and scores must have the same length.")
    if truth.sum() == 0:
        return float("nan")

    order = np.argsort(-values, kind="stable")
    ranked = truth[order]
    precision_at_k = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)

    return float(precision_at_k[ranked == 1].sum() / truth.sum())


def evaluate_detection(y_true, y_pred, scores=None) -> dict:
    """Compute every metric in one call; score-based metrics need `scores`."""
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    range_precision, range_recall, range_f1 = range_scores(y_true, y_pred)

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "point_adjusted_f1": point_adjusted_f1(y_true, y_pred),
        "range_precision": range_precision,
        "range_recall": range_recall,
        "range_f1": range_f1,
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "detection_delay": detection_delay(y_true, y_pred),
    }

    if scores is not None:
        result["roc_auc"] = roc_auc(y_true, scores)
        result["average_precision"] = average_precision(y_true, scores)

    return result
