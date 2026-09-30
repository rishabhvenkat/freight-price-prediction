# Freight Price Prediction System

A sanitized portfolio reconstruction of an ML-based freight price prediction system developed from professional project experience.

> **Portfolio note:** This repository contains an independently implemented reconstruction. Proprietary company code, production data, internal configurations, infrastructure details, and generated production artifacts are not included.

## Overview

The system predicts freight prices using a hybrid approach combining:

- Historical shipment-based pricing
- Gradient-boosted machine learning
- Route and distance features
- Quantity and distance interactions
- What-if scenario simulation
- SHAP-based explainability
- FastAPI inference serving

## Architecture

```text
Historical Shipment Data
          |
          v
   Data Quality Checks
          |
          v
    Feature Engineering
       /          \
      v            v
Historical        ML Model
Algorithm           |
       \            /
        \          /
         v        v
       Hybrid Prediction
             |
      +------+------+
      |      |      |
      v      v      v
     API  Scenarios SHAP
````

## Machine Learning

The portfolio implementation uses `HistGradientBoostingRegressor` with a chronological train/test split.

Evaluation on the included synthetic dataset:

| Metric           | Result |
| ---------------- | -----: |
| Training samples |  1,867 |
| Test samples     |    467 |
| MAE              |  24.33 |
| RMSE             |  31.04 |
| MAPE             |  5.00% |

These metrics are from the synthetic portfolio dataset and are **not production metrics**.

## Historical Algorithm

The historical component searches recent shipments for:

* Matching origin and destination
* Similar shipment quantity
* Recent historical observations
* Quantity similarity

Qualifying observations are combined into a weighted historical estimate.

The historical estimate can then be combined with the ML prediction through the hybrid prediction layer.

## Scenario Simulation

The system supports what-if analysis such as:

* Increasing shipment quantity
* Changing route distance
* Comparing predictions against a baseline

## Explainability

SHAP is used to inspect feature contributions for individual predictions.

Example features include:

* Quantity
* Distance × quantity interaction
* Weekday
* Week of year
* Month

## API

The project exposes a FastAPI service.

### Endpoints

* `GET /health`
* `GET /model/metrics`
* `POST /predict`

Example request:

```json
{
  "origin": "WH_02",
  "destination": "CITY_01",
  "distance_km": 600,
  "quantity_tons": 10,
  "order_date": "2025-01-15"
}
```

Example response:

```json
{
  "predicted_freight": 325.93,
  "model": "HistGradientBoostingRegressor"
}
```

## Project Structure

```text
predictor-server/
├── api/
│   └── app.py
├── data/
│   └── synthetic_freight_data.csv
├── src/
│   ├── preprocessing.py
│   ├── algorithm.py
│   ├── model.py
│   ├── hybrid_predictor.py
│   ├── simulator.py
│   └── explainability.py
├── notebooks/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Running Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api.app:app --reload --port 8001
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "WH_02",
    "destination": "CITY_01",
    "distance_km": 600,
    "quantity_tons": 10,
    "order_date": "2025-01-15"
  }'
```

## Engineering Focus

This project demonstrates:

* Time-aware ML evaluation
* Data-quality validation
* Feature engineering
* Historical similarity-based prediction
* Hybrid ML and algorithmic inference
* Scenario analysis
* Model explainability
* REST API serving

## Disclaimer

All shipment data included in this repository is synthetic. Implementation details have been independently reconstructed for public portfolio use, with proprietary company code, data, configurations, and infrastructure excluded.

````