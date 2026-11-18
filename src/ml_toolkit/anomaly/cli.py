"""Command-line interface for anomaly detection and benchmarking.

Run with `python -m ml_toolkit.anomaly.cli --help`. To expose it from the main CLI, add
`app.add_typer(anomaly_app, name="anomaly")` where the main Typer app is defined.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import typer

from .api import detect_anomalies
from .benchmark import AnomalyBenchmark
from .detectors import DETECTORS
from .injection import ANOMALY_TYPES
from .report import format_anomaly_report, results_to_json
from .thresholds import THRESHOLD_METHODS

app = typer.Typer(help="Detect and benchmark anomalies in time series.")


def parse_values(text: str) -> np.ndarray:
    """Parse numbers separated by whitespace, commas or semicolons; 'nan' means missing."""
    tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]

    if not tokens:
        raise ValueError("No values found in the input.")

    try:
        return np.array([float(token) for token in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Input contains a value that is not a number: {exc}") from exc


def detection_summary(values, method, threshold, train_fraction, period) -> dict:
    """Run detection and return a JSON-friendly summary."""
    kwargs = {"period": period} if method == "seasonal" else {}
    result = detect_anomalies(
        values, method=method, threshold=threshold, train_fraction=train_fraction, **kwargs
    )

    return {
        "method": result["method"],
        "threshold": result["threshold"],
        "points": int(len(result["scores"])),
        "anomalous_points": len(result["indices"]),
        "indices": result["indices"],
        "segments": [[int(start), int(stop)] for start, stop in result["segments"]],
    }


@app.command()
def detect(
    path: Path = typer.Argument(None, help="File with values; reads stdin when omitted."),
    method: str = typer.Option("rolling_mad", help=f"One of: {', '.join(DETECTORS)}."),
    threshold: float = typer.Option(None, help="Score threshold; defaults to the detector's own."),
    train_fraction: float = typer.Option(1.0, min=0.01, max=1.0, help="Leading share used for fitting."),
    period: int = typer.Option(24, min=2, help="Seasonal period (seasonal detector only)."),
):
    """Flag anomalies in a series of numbers."""
    text = path.read_text() if path is not None else sys.stdin.read()

    try:
        values = parse_values(text)
        summary = detection_summary(values, method, threshold, train_fraction, period)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(json.dumps(summary, indent=2))


@app.command()
def benchmark(
    n: int = typer.Option(600, min=48, help="Series length."),
    period: int = typer.Option(24, min=2),
    noise: list[float] = typer.Option([0.1, 0.3], help="Noise level; repeat the option for more."),
    kinds: list[str] = typer.Option(list(ANOMALY_TYPES), "--kind", help="Anomaly type; repeatable."),
    threshold_method: str = typer.Option("mad", help=f"One of: {', '.join(THRESHOLD_METHODS)}."),
    seed: int = typer.Option(0),
    output_format: str = typer.Option("markdown", "--format", help="markdown or json."),
):
    """Compare all detectors on synthetic anomalies."""
    if output_format not in ("markdown", "json"):
        raise typer.BadParameter("format must be 'markdown' or 'json'")

    try:
        bench = AnomalyBenchmark(
            anomaly_types=kinds,
            noise_levels=noise,
            n=n,
            period=period,
            threshold_method=threshold_method,
            random_state=seed,
        )
        results = bench.run()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(format_anomaly_report(results) if output_format == "markdown" else results_to_json(results))


if __name__ == "__main__":
    app()
