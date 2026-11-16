"""Statistical comparison utilities."""

import numpy as np
from scipy.stats import wilcoxon, ttest_rel


class StatisticalComparator:
    """Compare paired model evaluation results."""

    @staticmethod
    def validate(first, second):
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)

        if first.ndim != 1 or second.ndim != 1:
            raise ValueError("Inputs must be one-dimensional.")

        if len(first) != len(second):
            raise ValueError("Inputs must have equal lengths.")

        if len(first) < 2:
            raise ValueError("At least two paired observations are required.")

        if not np.all(np.isfinite(first)):
            raise ValueError("First input contains invalid values.")

        if not np.all(np.isfinite(second)):
            raise ValueError("Second input contains invalid values.")

        return first, second

    def compare_means(self, first, second):
        first, second = self.validate(first, second)

        statistic, p_value = ttest_rel(first, second)

        return {
            "test": "paired_t_test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }

    def compare_ranks(self, first, second):
        first, second = self.validate(first, second)

        if np.all(first == second):
            return {
                "test": "wilcoxon",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
            }

        statistic, p_value = wilcoxon(first, second)

        return {
            "test": "wilcoxon",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }