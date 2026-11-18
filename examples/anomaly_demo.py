"""Compare anomaly detectors on synthetic seasonal data and flag anomalies in one series."""

from ml_toolkit.anomaly import (
    AnomalyBenchmark,
    detect_anomalies,
    format_anomaly_report,
    inject_anomalies,
    make_seasonal_series,
)

# 1. Benchmark every detector across all anomaly types.
benchmark = AnomalyBenchmark(noise_levels=(0.1, 0.3), threshold_method="mad", random_state=42)
print(format_anomaly_report(benchmark.run()))

# 2. Detect anomalies in a single series.
series = make_seasonal_series(600, period=24, noise=0.1, random_state=7)
corrupted, labels = inject_anomalies(series, kind="level_shift", count=4, random_state=7)

result = detect_anomalies(corrupted, method="seasonal", train_fraction=0.3, period=24)

print()
print(f"True anomalous points: {int(labels.sum())}")
print(f"Flagged segments (start, stop): {result['segments']}")
