"""FastAPI application for local inference."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from local_ml_toolkit.model import DEFAULT_MODEL, LocalTextClassifier

app = FastAPI(
    title="Local ML Toolkit",
    description="A minimal API for local pretrained text classification.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)


@lru_cache(maxsize=1)
def get_model() -> LocalTextClassifier:
    return LocalTextClassifier(model_id=DEFAULT_MODEL, device="auto")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model.cache_info().currsize > 0}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        return {"predictions": get_model().predict_many(request.texts)}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
