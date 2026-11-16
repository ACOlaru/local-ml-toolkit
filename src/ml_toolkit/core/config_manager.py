"""Manage reusable benchmark configurations."""

import json
from pathlib import Path


class BenchmarkConfigManager:
    """Create, save, load and list benchmark configurations."""

    def __init__(self, directory="configs"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_config(self, name, configuration):
        if not name or Path(name).name != name:
            raise ValueError("Configuration name must be a simple filename.")

        path = self.directory / f"{name}.json"

        with path.open("w", encoding="utf-8") as file:
            json.dump(configuration, file, indent=4)

        return path

    def load_config(self, name):
        if not name or Path(name).name != name:
            raise ValueError("Invalid configuration name.")

        path = self.directory / f"{name}.json"

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_configs(self):
        return sorted(
            path.stem
            for path in self.directory.glob("*.json")
        )

    def delete_config(self, name):
        if not name or Path(name).name != name:
            raise ValueError("Invalid configuration name.")

        path = self.directory / f"{name}.json"

        if not path.exists():
            raise FileNotFoundError(path)

        path.unlink()