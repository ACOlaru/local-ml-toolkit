"""Command-line interface."""

from __future__ import annotations

import json
import sys
import time
from typing import Optional

import psutil
import typer

from ml_toolkit.time_series.models import DEFAULT_MODEL, LocalTextClassifier

app = typer.Typer(help="Run pretrained models locally and benchmark inference.")


@app.command()
def predict(
    texts: list[str] = typer.Argument(..., help="One or more input texts."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Hugging Face model ID."),
    device: str = typer.Option("auto", help="auto, cpu, or cuda."),
    batch_size: int = typer.Option(8, min=1),
):
    """Run one or more predictions."""
    classifier = LocalTextClassifier(model_id=model, device=device, batch_size=batch_size)
    results = classifier.predict_many(texts)
    typer.echo(json.dumps(results, indent=2))


@app.command()
def stream(
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    device: str = typer.Option("auto", help="auto, cpu, or cuda."),
    batch_size: int = typer.Option(8, min=1),
):
    """Read JSONL records from stdin; each line must contain a `text` field."""
    classifier = LocalTextClassifier(model_id=model, device=device, batch_size=batch_size)
    pending: list[str] = []

    def flush() -> None:
        for prediction in classifier.predict_many(pending):
            typer.echo(json.dumps(prediction))
        pending.clear()

    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            pending.append(record["text"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise typer.BadParameter(f"Invalid JSONL record on line {line_number}: {exc}") from exc
        if len(pending) >= batch_size:
            flush()
    flush()


@app.command()
def benchmark(
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    device: str = typer.Option("auto", help="auto, cpu, or cuda."),
    samples: int = typer.Option(64, min=1),
    batch_size: int = typer.Option(8, min=1),
    warmup: int = typer.Option(2, min=0),
    repeats: int = typer.Option(5, min=1),
):
    """Measure simple local inference latency and throughput."""
    if samples < 1 or batch_size < 1 or repeats < 1:
        raise typer.BadParameter("samples, batch-size, and repeats must be positive")

    classifier = LocalTextClassifier(model_id=model, device=device, batch_size=batch_size)
    texts = ["A short example sentence for local model benchmarking."] * samples

    for _ in range(warmup):
        classifier.predict_many(texts[: min(batch_size, samples)])

    if classifier.device_name == "cuda":
        classifier._torch.cuda.synchronize()
        classifier._torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    rss_before = process.memory_info().rss
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        for offset in range(0, len(texts), batch_size):
            classifier.predict_many(texts[offset : offset + batch_size])
        if classifier.device_name == "cuda":
            classifier._torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)

    total_seconds = sum(durations) / len(durations)
    result = {
        "model": model,
        "device": classifier.device_name,
        "samples_per_repeat": samples,
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "mean_seconds": round(total_seconds, 4),
        "mean_latency_ms_per_sample": round(total_seconds * 1000 / samples, 3),
        "throughput_samples_per_second": round(samples / total_seconds, 2),
        "process_rss_delta_mb": round(
            (process.memory_info().rss - rss_before) / (1024 * 1024), 2
        ),
        "peak_cuda_allocated_mb": (
            round(classifier._torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            if classifier.device_name == "cuda"
            else None
        ),
        "note": "Indicative local measurement; record hardware and software versions.",
    }
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
