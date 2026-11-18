import json

import numpy as np
import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner  # noqa: E402

from ml_toolkit.anomaly.cli import app, detection_summary, parse_values  # noqa: E402


def spike_series():
    values = np.random.default_rng(0).normal(0, 1, size=200)
    values[120] = 25
    return values


def test_parse_values_accepts_mixed_separators():
    assert parse_values("1, 2;3\n4").tolist() == [1.0, 2.0, 3.0, 4.0]


def test_parse_values_keeps_nan():
    assert np.isnan(parse_values("1 nan 3")[1])


def test_parse_values_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_values("   ")
    with pytest.raises(ValueError):
        parse_values("1 two 3")


def test_detection_summary_reports_spike():
    summary = detection_summary(spike_series(), "zscore", None, 1.0, 24)

    assert 120 in summary["indices"]
    assert summary["points"] == 200
    assert summary["anomalous_points"] == len(summary["indices"])


def test_detect_command_prints_json(tmp_path):
    path = tmp_path / "values.txt"
    path.write_text("\n".join(str(value) for value in spike_series()))

    result = CliRunner().invoke(app, ["detect", str(path), "--method", "zscore"])

    assert result.exit_code == 0
    assert 120 in json.loads(result.output)["indices"]


def test_detect_command_rejects_unknown_method(tmp_path):
    path = tmp_path / "values.txt"
    path.write_text("1 2 3 4 5 6")

    result = CliRunner().invoke(app, ["detect", str(path), "--method", "nope"])

    assert result.exit_code != 0
