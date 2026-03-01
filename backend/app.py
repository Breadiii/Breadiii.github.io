from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = Path(__file__).with_name("pressure_model.pkl")
CAT_FEATURES = [
    "LOCATION_POSTAL_CODE",
    "SECTOR",
    "OVERNIGHT_SERVICE_TYPE",
    "PROGRAM_MODEL",
    "PROGRAM_AREA",
    "CAPACITY_TYPE",
]
NUM_FEATURES = [
    "ACTUAL_CAPACITY",
    "lat",
    "lon",
    "dow",
    "month",
    "day",
]

# allow your GitHub Pages site to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://breadiii.github.io",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    artifact, model_error = get_model_artifact()
    return {"ok": True, "model_loaded": artifact is not None, "model_error": model_error}


class PredictReq(BaseModel):
    LOCATION_POSTAL_CODE: str
    SECTOR: str | None = None
    OVERNIGHT_SERVICE_TYPE: str | None = None
    PROGRAM_MODEL: str | None = None
    PROGRAM_AREA: str | None = None
    CAPACITY_TYPE: str | None = None
    ACTUAL_CAPACITY: float
    lat: float
    lon: float
    OCCUPANCY_DATE: str


@lru_cache(maxsize=1)
def get_model_artifact() -> tuple[dict | None, str | None]:
    if not MODEL_PATH.exists():
        return None, f"Model artifact not found at {MODEL_PATH}"

    try:
        with MODEL_PATH.open("rb") as fh:
            artifact = pickle.load(fh)
        if "model" not in artifact:
            raise ValueError("Artifact is missing the `model` key")
        return artifact, None
    except Exception as exc:
        return None, str(exc)


def _build_prediction_frame(req: PredictReq) -> pd.DataFrame:
    occ_date = pd.to_datetime(req.OCCUPANCY_DATE, errors="coerce")
    if pd.isna(occ_date):
        raise ValueError("OCCUPANCY_DATE must be a valid date string")

    row = {
        "LOCATION_POSTAL_CODE": req.LOCATION_POSTAL_CODE,
        "SECTOR": req.SECTOR,
        "OVERNIGHT_SERVICE_TYPE": req.OVERNIGHT_SERVICE_TYPE,
        "PROGRAM_MODEL": req.PROGRAM_MODEL,
        "PROGRAM_AREA": req.PROGRAM_AREA,
        "CAPACITY_TYPE": req.CAPACITY_TYPE,
        "ACTUAL_CAPACITY": req.ACTUAL_CAPACITY,
        "lat": req.lat,
        "lon": req.lon,
        "dow": occ_date.dayofweek,
        "month": occ_date.month,
        "day": occ_date.day,
    }
    return pd.DataFrame([row], columns=CAT_FEATURES + NUM_FEATURES)


@app.post("/predict")
def predict(req: PredictReq):
    artifact, model_error = get_model_artifact()
    if artifact is None:
        pressure = max(0.0, min(1.0, req.ACTUAL_CAPACITY / 100.0))
        return {
            "pred_pressure": round(pressure, 4),
            "model_status": "fallback",
            "model_error": model_error,
        }

    features = _build_prediction_frame(req)
    pipe = artifact["model"]
    pred_logit = float(pipe.predict(features)[0])
    pred = 1.0 / (1.0 + np.exp(-pred_logit))
    pred = max(0.0, min(1.0, pred))
    return {"pred_pressure": round(float(pred), 4), "model_status": "artifact"}


@app.get("/")
def root():
    return {"ok": True, "message": "API is running. Try /docs or /predict"}
