from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# allow your GitHub Pages site to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://breadiii.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

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

@app.post("/predict")
def predict(req: PredictReq):
    # TODO: replace with real model prediction
    return {"pred_pressure": 0.5}
