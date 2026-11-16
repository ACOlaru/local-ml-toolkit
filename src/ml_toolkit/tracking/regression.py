"""Detect performance regressions between benchmark runs."""


class RegressionDetector:
    """Compare current benchmark results against a baseline."""

    def __init__(self, tolerance=0.05):
        if tolerance < 0:
            raise ValueError("Tolerance cannot be negative.")

        self.tolerance = tolerance

    def compare_metric(self, baseline, current, higher_is_better=False):
        if baseline <= 0:
            raise ValueError("Baseline must be positive.")

        change = (current - baseline) / abs(baseline)

        if higher_is_better:
            regression = change < -self.tolerance
        else:
            regression = change > self.tolerance

        return {
            "baseline": baseline,
            "current": current,
            "change_percent": round(change * 100, 2),
            "regression_detected": regression,
        }

    def compare_results(self, baseline, current):
        metrics = {}

        for metric in baseline:
            if metric not in current:
                continue

            metrics[metric] = self.compare_metric(
                baseline[metric],
                current[metric],
                higher_is_better=metric in {
                    "accuracy",
                    "precision",
                    "recall",
                    "f1"
                }
            )

        return {
            "metrics": metrics,
            "has_regression": any(
                result["regression_detected"]
                for result in metrics.values()
            )
        }