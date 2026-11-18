import numpy as np
import pytest

from ml_toolkit.anomaly.metrics import (
    average_precision,
    confusion_counts,
    detection_delay,
    evaluate_detection,
    false_alarm_rate,
    point_adjust,
    point_adjusted_f1,
    precision_recall_f1,
    range_scores,
    roc_auc,
    segments,
)

TRUTH = [0, 1, 1, 1, 0, 1, 1]


def test_confusion_counts():
    counts = confusion_counts([1, 1, 0, 0], [1, 0, 1, 0])

    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_precision_recall_f1():
    precision, recall, f1 = precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])

    assert (precision, recall, f1) == (0.5, 0.5, 0.5)


def test_precision_recall_f1_without_predictions_is_zero():
    assert precision_recall_f1([1, 1, 0], [0, 0, 0]) == (0.0, 0.0, 0.0)


def test_false_alarm_rate():
    assert false_alarm_rate([0, 0, 0, 0], [1, 0, 0, 0]) == 0.25


def test_segments():
    assert segments([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    assert segments([0, 0, 0]) == []
    assert segments([1, 1, 1]) == [(0, 3)]


def test_point_adjust_fills_detected_segments_only():
    adjusted = point_adjust(TRUTH, [0, 0, 1, 0, 0, 0, 0])

    assert adjusted.tolist() == [0, 1, 1, 1, 0, 0, 0]


def test_point_adjusted_f1():
    assert point_adjusted_f1(TRUTH, [0, 0, 1, 0, 0, 0, 0]) == pytest.approx(0.75)


def test_range_scores():
    precision, recall, f1 = range_scores(TRUTH, [0, 0, 1, 0, 1, 0, 0])

    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_range_scores_without_predictions():
    assert range_scores(TRUTH, [0] * 7) == (0.0, 0.0, 0.0)


def test_detection_delay():
    assert detection_delay(TRUTH, [0, 0, 1, 0, 0, 0, 0]) == 1.0
    assert np.isnan(detection_delay(TRUTH, [0] * 7))


def test_roc_auc_extremes_and_ties():
    labels = [0, 0, 1, 1]

    assert roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc(labels, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_average_precision():
    assert average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(5 / 6)
    assert np.isnan(average_precision([0, 0], [0.1, 0.2]))


def test_evaluate_detection_keys():
    labels = [0, 1, 1, 0, 0, 1]
    flags = [0, 1, 0, 0, 1, 1]

    without_scores = evaluate_detection(labels, flags)
    with_scores = evaluate_detection(labels, flags, scores=[0.1, 0.9, 0.4, 0.2, 0.6, 0.8])

    assert "f1" in without_scores
    assert "roc_auc" not in without_scores
    assert {"roc_auc", "average_precision", "range_f1"} <= set(with_scores)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        confusion_counts([0, 1], [0, 1, 1])
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1, 0.2, 0.3])
