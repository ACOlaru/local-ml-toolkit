"""Track machine learning experiments."""

import json
from pathlib import Path
from datetime import datetime


class ExperimentTracker:
    """Save and retrieve benchmark experiments."""

    def __init__(self, storage_path="experiments.json"):
        self.storage_path = Path(storage_path)
        self.experiments = self._load()

    def _load(self):
        if not self.storage_path.exists():
            return []

        with self.storage_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def save(self):
        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.storage_path.open("w", encoding="utf-8") as file:
            json.dump(self.experiments, file, indent=4)

    def add_experiment(
        self,
        model_name,
        metrics,
        configuration=None,
        notes=""
    ):
        experiment = {
            "id": len(self.experiments) + 1,
            "model": model_name,
            "metrics": metrics,
            "configuration": configuration or {},
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }

        self.experiments.append(experiment)
        self.save()

        return experiment

    def get_all(self):
        return list(self.experiments)

    def get_by_model(self, model_name):
        return [
            experiment
            for experiment in self.experiments
            if experiment["model"] == model_name
        ]

    def clear(self):
        self.experiments = []
        self.save()