"""Readable and machine-readable anomaly benchmark reports."""

from __future__ import annotations

import json
import math

from .benchmark import best_by_group, summarize_results


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def format_anomaly_report(results) -> str:
    """Format benchmark results as Markdown."""
    if not results:
        raise ValueError("No benchmark results supplied.")

    overall = summarize_results(results, by="detector")
    lines = [
        "# Anomaly Detection Benchmark Report",
        "",
        "## Overall ranking (mean over all cases)",
        "",
        "| Detector | F1 | Point-adj. F1 | Range F1 | ROC AUC | AP | Delay | False alarms | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in overall:
        lines.append(
            f"| {row['detector']} "
            f"| {_fmt(row['f1'])} "
            f"| {_fmt(row['point_adjusted_f1'])} "
            f"| {_fmt(row['range_f1'])} "
            f"| {_fmt(row['roc_auc'])} "
            f"| {_fmt(row['average_precision'])} "
            f"| {_fmt(row['detection_delay'], 1)} "
            f"| {_fmt(row['false_alarm_rate'])} "
            f"| {_fmt(row['runtime_seconds'], 4)} |"
        )

    by_type = summarize_results(results, by="anomaly_type")
    winners = best_by_group(results, by="anomaly_type")
    lines.extend([
        "",
        "## Best detector per anomaly type",
        "",
        "| Anomaly type | Best detector (by F1) | Mean F1 across detectors |",
        "|---|---|---:|",
    ])

    for row in sorted(by_type, key=lambda item: item["anomaly_type"]):
        lines.append(f"| {row['anomaly_type']} | {winners[row['anomaly_type']]} | {_fmt(row['f1'])} |")

    lines.extend([
        "",
        f"Best detector overall by F1: **{overall[0]['detector']}**",
        "",
        "Thresholds are fitted on a clean training window unless `best_f1` was requested.",
    ])

    return "\n".join(lines)


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def results_to_json(results, indent=2) -> str:
    """Serialise benchmark rows as JSON, turning NaN into null."""
    return json.dumps([{key: _clean(value) for key, value in row.items()} for row in results], indent=indent)
