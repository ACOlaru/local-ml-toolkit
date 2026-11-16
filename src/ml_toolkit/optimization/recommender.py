"""Recommend models using measured benchmark results."""


class ModelRecommender:
    """Rank models using user-defined performance priorities."""

    def __init__(
        self,
        accuracy_weight=0.5,
        speed_weight=0.25,
        memory_weight=0.25
    ):
        weights = [
            accuracy_weight,
            speed_weight,
            memory_weight
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative.")

        if sum(weights) <= 0:
            raise ValueError("At least one weight must be positive.")

        total = sum(weights)

        self.weights = [
            weight / total for weight in weights
        ]

    def recommend(self, results):
        if not results:
            raise ValueError("No benchmark results provided.")

        required = {
            "model",
            "accuracy",
            "runtime_seconds",
            "memory_mb"
        }

        for result in results:
            if not required.issubset(result):
                raise ValueError("A benchmark result is missing required fields.")

            if result["runtime_seconds"] <= 0 or result["memory_mb"] <= 0:
                raise ValueError("Runtime and memory must be positive.")

        max_runtime = max(r["runtime_seconds"] for r in results)
        max_memory = max(r["memory_mb"] for r in results)

        min_accuracy = min(r["accuracy"] for r in results)
        max_accuracy = max(r["accuracy"] for r in results)

        recommendations = []

        for result in results:
            if max_accuracy == min_accuracy:
                accuracy_score = 1.0
            else:
                accuracy_score = (
                    result["accuracy"] - min_accuracy
                ) / (max_accuracy - min_accuracy)

            speed_score = 1 - (
                result["runtime_seconds"] / max_runtime
            )

            memory_score = 1 - (
                result["memory_mb"] / max_memory
            )

            score = (
                accuracy_score * self.weights[0]
                + speed_score * self.weights[1]
                + memory_score * self.weights[2]
            )

            recommendations.append({
                **result,
                "recommendation_score": round(score, 4)
            })

        return sorted(
            recommendations,
            key=lambda item: item["recommendation_score"],
            reverse=True
        )