"""BAirOps FastAPI application: REST API + dashboard."""
import os
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.service import Service

SERVICE = Service()
STATIC = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    SERVICE.start(background=True)          # load production model, then refresh live data every hour
    yield


app = FastAPI(title="BAirOps API", version="1.0.0", lifespan=lifespan,
              description="Near real-time 24-h AQI forecasts for Bangladesh from a stacked ensemble "
                          "(Random Forest, XGBoost, LightGBM, CatBoost) served from the MLflow model registry.")


class PredictRequest(BaseModel):
    rows: List[Dict[str, float]] = Field(..., description="One dict of feature values per forecast (see /model-info)")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/health")
def health():
    return SERVICE.health()


@app.get("/model-info")
def model_info():
    if SERVICE.model is None:
        raise HTTPException(503, "model is loading")
    return dict(SERVICE.model.info(), features=SERVICE.model.features)


@app.get("/cities")
def cities():
    return SERVICE.cities()


@app.get("/forecast")
def forecast(city: str = Query(..., description="City name, e.g. Dhaka")):
    try:
        f = SERVICE.forecast(city)
    except KeyError:
        raise HTTPException(404, f"unknown city '{city}'; see /cities")
    if f is None:
        raise HTTPException(503, "forecasts are being prepared, retry in a minute")
    return f


@app.get("/forecast/all")
def forecast_all():
    if not SERVICE.ready():
        raise HTTPException(503, "forecasts are being prepared, retry in a minute")
    return SERVICE.all_forecasts()


@app.post("/predict")
def predict(req: PredictRequest):
    if SERVICE.model is None:
        raise HTTPException(503, "model is loading")
    if not req.rows:
        raise HTTPException(422, "rows must not be empty")
    try:
        return SERVICE.predict_rows(req.rows)
    except ValueError as e:
        raise HTTPException(422, str(e))
