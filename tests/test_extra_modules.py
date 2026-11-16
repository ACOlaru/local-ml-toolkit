import pandas as pd
import pytest

from ml_toolkit.data_quality.validator import DatasetValidator
from ml_toolkit.core.hardware import HardwareInspector
from ml_toolkit.core.config_manager import BenchmarkConfigManager
from ml_toolkit.optimization.recommender import ModelRecommender
from ml_toolkit.tracking.regression import RegressionDetector


def test_dataset_validator():
    data = pd.DataFrame({
        "age": [20, 30, 30],
        "score": [80, None, 90]
    })

    validator = DatasetValidator(data)

    assert validator.check_duplicates() == 0
    assert validator.check_missing_values()["score"] == 1


def test_hardware_inspector():
    inspector = HardwareInspector()
    report = inspector.get_system_report()

    assert "cpu" in report
    assert "memory" in report


def test_config_manager(tmp_path):
    manager = BenchmarkConfigManager(tmp_path)

    manager.save_config("test", {"batch_size": 16})

    assert manager.load_config("test")["batch_size"] == 16


def test_recommender():
    results = [
        {
            "model": "A",
            "accuracy": 0.95,
            "runtime_seconds": 2,
            "memory_mb": 500
        },
        {
            "model": "B",
            "accuracy": 0.90,
            "runtime_seconds": 1,
            "memory_mb": 200
        }
    ]

    ranked = ModelRecommender().recommend(results)

    assert len(ranked) == 2
    assert ranked[0]["recommendation_score"] >= ranked[1]["recommendation_score"]


def test_regression_detector():
    detector = RegressionDetector(tolerance=0.05)

    result = detector.compare_metric(1.0, 1.2)

    assert result["regression_detected"] is True


def test_config_missing_file(tmp_path):
    manager = BenchmarkConfigManager(tmp_path)

    with pytest.raises(FileNotFoundError):
        manager.load_config("missing")