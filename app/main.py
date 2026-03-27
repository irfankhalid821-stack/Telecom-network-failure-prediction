from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Telecom Failure Prediction API")

MODEL_PATH = Path(file).resolve().parents[1] / "model" / "model.pkl"
model = None


class PredictionRequest(BaseModel):
    signal_strength: int = Field(..., ge=-120, le=-30)
    temperature: int = Field(..., ge=-20, le=80)
    humidity: int = Field(..., ge=0, le=100)
    network_load: int = Field(..., ge=0, le=100)


@app.on_event("startup")
def load_model() -> None:
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Telecom Failure Prediction API"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded" if model is not None else "missing"}


@app.post("/predict")
def predict(data: PredictionRequest) -> dict[str, int]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run python model/train_model.py first.",
        )

    try:
        values = np.array(
            [
                data.signal_strength,
                data.temperature,
                data.humidity,
                data.network_load,
            ]
        ).reshape(1, -1)
        prediction = model.predict(values)[0]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {"prediction": int(prediction)}
