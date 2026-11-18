import json
import math

import pytest

from ml_toolkit.anomaly.api import detect_anomalies
from ml_toolkit.anomaly.benchmark import AnomalyBenchmark, best_by_group, summarize_results
from ml_toolkit.anomaly.report import format_anomaly_report, results_to_json

SMALL = dict(
    anomaly_types=("spike", "level_shift"),
    noise_levels=(0.1,),
    n=240,
    period=12,
)


@pytest.fixture(scope="module")
def results():
    return AnomalyBenchmark(**SMALL).run()


def test_run_returns_one_row_per_case(results):
    assert len(results) == 2 * 1 * 6
    assert {row["detector"] for row in results} == {
        "zscore", "iqr", "rolling_mad", "ewma", "seasonal", "isolation_forest",
    }
    assert {"f1", "roc_auc", "runtime_seconds", "anomaly_type", "noise"} <= set(results[0])


def test_run_is_reproducible(results):
    again = AnomalyBenchmark(**SMALL).run()

    assert [row["f1"] for row in results] == [row["f1"] for row in again]


def test_oracle_threshold_is_an_upper_bound(results):
    oracle = AnomalyBenchmark(threshold_method="best_f1", **SMALL).run()

    for normal, tuned in zip(results, oracle):
        assert (normal["detector"], normal["anomaly_type"]) == (tuned["detector"], tuned["anomaly_type"])
        assert tuned["f1"] >= normal["f1"] - 1e-9


def test_seasonal_detector_finds_spikes_with_a_good_threshold():
    oracle = AnomalyBenchmark(threshold_method="best_f1", **SMALL).run()
    spikes = [row for row in oracle if row["anomaly_type"] == "spike"]

    assert max(row["f1"] for row in spikes) >= 0.8


def test_benchmark_validation():
    with pytest.raises(ValueError):
        AnomalyBenchmark(train_fraction=0.95)
    with pytest.raises(ValueError):
        AnomalyBenchmark(n=20, period=24)
    with pytest.raises(ValueError):
        AnomalyBenchmark(anomaly_types=("nonsense",))


def test_summarize_results_sorts_by_f1(results):
    summary = summarize_results(results)

    assert [row["f1"] for row in summary] == sorted((row["f1"] for row in summary), reverse=True)
    assert all(row["cases"] == 2 for row in summary)


def test_summarize_results_ignores_nan():
    rows = [
        {"detector": "a", "f1": 0.5, "detection_delay": float("nan")},
        {"detector": "a", "f1": 0.7, "detection_delay": 3.0},
    ]

    summary = summarize_results(rows)[0]

    assert summary["f1"] == pytest.approx(0.6)
    assert summary["detection_delay"] == 3.0
    assert math.isnan(summary["roc_auc"])


def test_summarize_results_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize_results([])


def test_best_by_group_covers_every_type(results):
    winners = best_by_group(results, by="anomaly_type")

    assert set(winners) == {"spike", "level_shift"}


def test_format_report_contains_expected_sections(results):
    report = format_anomaly_report(results)

    assert "# Anomaly Detection Benchmark Report" in report
    assert "Best detector per anomaly type" in report
    assert "isolation_forest" in report


def test_format_report_rejects_empty_input():
    with pytest.raises(ValueError):
        format_anomaly_report([])


def test_results_to_json_turns_nan_into_null():
    payload = json.loads(results_to_json([{"a": float("nan"), "b": 1}]))

    assert payload == [{"a": None, "b": 1}]


def test_detect_anomalies_finds_a_spike():
    import numpy as np

    values = np.random.default_rng(0).normal(0, 1, size=300)
    values[150] = 20

    result = detect_anomalies(values, method="zscore")

    assert 150 in result["indices"]
    assert result["segments"] == [(150, 151)]
    assert {"method", "threshold", "scores", "flags", "indices", "segments"} <= set(result)


def test_detect_anomalies_validation():
    with pytest.raises(ValueError):
        detect_anomalies([1.0, 2.0, 3.0], train_fraction=0)
    with pytest.raises(ValueError):
        detect_anomalies([1.0, 2.0, 3.0], method="unknown")
