# Anomaly detection

The `ml_toolkit.anomaly` package adds unsupervised anomaly detection for univariate time
series, plus tools to benchmark detectors on synthetic data with known anomalies.

## Detectors

| Name | Idea | Default threshold |
|---|---|---:|
| `zscore` | Distance from the training mean in standard deviations | 3.0 |
| `iqr` | Distance from the median in interquartile ranges (Tukey fence at 2.0) | 2.0 |
| `rolling_mad` | Robust z-score against the preceding window | 3.5 |
| `ewma` | One-step-ahead residual of an exponentially weighted average | 3.5 |
| `seasonal` | Residual after removing trend and a seasonal profile | 4.0 |
| `isolation_forest` | Isolation forest over sliding windows (scores in 0 to 1) | 0.6 |

All detectors share `fit`, `score`, `predict` and `fit_score`. Higher scores are more
anomalous.

## Quick start

```python
from ml_toolkit.anomaly import detect_anomalies

result = detect_anomalies(values, method="seasonal", train_fraction=0.3, period=24)
print(result["segments"])
```

## Benchmarking

```python
from ml_toolkit.anomaly import AnomalyBenchmark, format_anomaly_report

results = AnomalyBenchmark(noise_levels=(0.1, 0.3)).run()
print(format_anomaly_report(results))
```

Anomaly types: `spike`, `level_shift`, `variance_burst`, `contextual`, `collective`.
Thresholds are fitted on a clean training window (`mad`, `percentile` or `pot`). Use
`threshold_method="best_f1"` for an oracle upper bound that tunes on the labels.

## Metrics

Point-wise precision, recall and F1, point-adjusted F1, range-based (event) F1, detection
delay, false-alarm rate, ROC AUC and average precision.

## Command line

```
python -m ml_toolkit.anomaly.cli detect values.txt --method zscore
python -m ml_toolkit.anomaly.cli benchmark --format markdown
```
