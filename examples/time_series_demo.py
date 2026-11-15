import numpy as np

from ml_toolkit.time_series import (
    TimeSeriesBenchmark,
    NaiveForecaster,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
)

from ml_toolkit.time_series.corruption import (
    add_noise,
    add_missing_values,
    fill_missing,
)

from ml_toolkit.time_series.report import format_report


# Create example time-series data
values = np.sin(np.linspace(0, 20, 200)) + 10

# Add controlled noise
values = add_noise(values, noise_level=0.1, random_state=42)

# Simulate missing observations and fill them
values = add_missing_values(values, missing_rate=0.05, random_state=42)
values = fill_missing(values, strategy="mean")

# Define benchmark models
models = [
    NaiveForecaster(),
    MovingAverageForecaster(window=5),
    ExponentialSmoothingForecaster(alpha=0.3),
]

# Run benchmark
benchmark = TimeSeriesBenchmark(models)
results = benchmark.run(values, test_size=30)

# Print report
print(format_report(results))