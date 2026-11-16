import os
from pathlib import Path

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))

model = mlflow.sklearn.load_model(str(MODEL_DIR))
MODEL_VERSION = (MODEL_DIR / "VERSION").read_text().strip() if (MODEL_DIR / "VERSION").exists() else "unknown"

app = FastAPI(title="Digits classifier")


class PredictRequest(BaseModel):
    # Une ou plusieurs images 8x8 aplaties (64 pixels, valeurs 0-16), comme sklearn load_digits
    instances: list[list[float]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[int]
    model_version: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    X = np.asarray(req.instances, dtype=float)
    preds = model.predict(X)
    return PredictResponse(predictions=[int(p) for p in preds], model_version=MODEL_VERSION)
