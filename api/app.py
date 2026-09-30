from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.preprocessing import load_data, clean_data, create_features
from src.model import FreightModel


app = FastAPI(
    title="Freight Price Prediction API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    origin: str
    destination: str
    distance_km: float
    quantity_tons: float
    order_date: str


# Train once when the API starts.
historical_data = clean_data(load_data())
historical_data = create_features(historical_data)

model = FreightModel()
model._build_preprocessor(historical_data)
metrics = model.fit(historical_data)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "HistGradientBoostingRegressor",
    }


@app.get("/model/metrics")
def model_metrics():
    return metrics


@app.post("/predict")
def predict(request: PredictionRequest):
    row = {
        "order_date": request.order_date,
        "origin": request.origin,
        "destination": request.destination,
        "distance_km": request.distance_km,
        "quantity_tons": request.quantity_tons,
        "freight": 0.0,
    }

    features = pd.DataFrame([row])
    features["order_date"] = pd.to_datetime(features["order_date"])
    features = create_features(features)

    prediction = float(model.predict(features)[0])

    return {
        "predicted_freight": round(prediction, 2),
        "model": "HistGradientBoostingRegressor",
    }
