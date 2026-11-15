"""Local Hugging Face inference wrapper."""

from __future__ import annotations

from typing import Any, Sequence

DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class LocalTextClassifier:
    """Load a text-classification model and run local inference.

    Model weights are downloaded from Hugging Face on first use and cached locally.
    Pass ``device="cpu"`` for CPU inference, ``device="cuda"`` for NVIDIA CUDA,
    or ``device="auto"`` to select CUDA when available.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_id = model_id
        self.batch_size = batch_size
        self.device_name = self._resolve_device(device)

        # Lazy import keeps importing the package lightweight and helps tests avoid
        # downloading model weights unless an instance is actually created.
        import torch
        from transformers import pipeline

        pipeline_device = 0 if self.device_name == "cuda" else -1
        self._pipeline = pipeline(
            task="text-classification",
            model=model_id,
            device=pipeline_device,
        )
        self._torch = torch

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch install")
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def predict(self, text: str) -> dict[str, Any]:
        """Classify one text string."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        output = self._pipeline(text, truncation=True)[0]
        return {
            "text": text,
            "label": str(output["label"]),
            "score": float(output["score"]),
            "model": self.model_id,
            "device": self.device_name,
        }

    def predict_many(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """Classify multiple strings in a single pipeline call."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("every item in texts must be a non-empty string")

        outputs = self._pipeline(
            list(texts),
            batch_size=self.batch_size,
            truncation=True,
        )
        return [
            {
                "text": text,
                "label": str(output["label"]),
                "score": float(output["score"]),
                "model": self.model_id,
                "device": self.device_name,
            }
            for text, output in zip(texts, outputs, strict=True)
        ]

    def stream(self, texts: Sequence[str]):
        """Yield predictions one by one while processing bounded-size batches."""
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            yield from self.predict_many(batch)
