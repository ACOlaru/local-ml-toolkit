"""Benchmark multiple forecasting models."""

from time import perf_counter

import numpy as np

from .metrics import evaluate_forecast


class TimeSeriesBenchmark:
    """Compare forecasting models on the same test period."""

    def __init__(self, models):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models

    def run(self, values, test_size):
        data = np.asarray(values, dtype=float)

        if data.ndim != 1:
            raise ValueError("Input data must be one-dimensional.")

        if not 0 < test_size < len(data):
            raise ValueError("test_size must be between 0 and data length.")

        if not np.all(np.isfinite(data)):
            raise ValueError("Input data must not contain missing values.")

        train = data[:-test_size]
        test = data[-test_size:]

        results = []

        for model in self.models:
            start = perf_counter()

            fitted_model = model.fit(train)
            predictions = fitted_model.predict(test_size)

            elapsed = perf_counter() - start

            metrics = evaluate_forecast(test, predictions)

            results.append({
                "model": model.name,
                "runtime_seconds": elapsed,
                "predictions": predictions.tolist(),
                **metrics,
            })

        return sorted(results, key=lambda result: result["mae"])

    @staticmethod
    def best_model(results):
        """Return the model with the lowest MAE."""
        if not results:
            raise ValueError("Results cannot be empty.")

        return min(results, key=lambda result: result["mae"])