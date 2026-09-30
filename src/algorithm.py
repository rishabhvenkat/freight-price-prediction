import pandas as pd
import numpy as np


def get_algorithm_candidate(
    historical_data,
    origin,
    destination,
    quantity_tons,
    prediction_date,
    lookback_days=60,
    quantity_tolerance=0.35,
    similarity_threshold=0.15,
):
    """
    Estimate freight from recent historical shipments on a similar route.

    This is a generic portfolio implementation of the historical
    matching algorithm used as one component of a hybrid predictor.
    """

    if historical_data.empty:
        return None

    df = historical_data.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    prediction_date = pd.Timestamp(prediction_date)

    route_data = df[
        (df["origin"] == origin)
        & (df["destination"] == destination)
        & (df["order_date"] < prediction_date)
    ].copy()

    if route_data.empty:
        return None

    cutoff = prediction_date - pd.Timedelta(days=lookback_days)

    route_data = route_data[
        route_data["order_date"] >= cutoff
    ].copy()

    if route_data.empty:
        return None

    min_quantity = quantity_tons * (1 - quantity_tolerance)
    max_quantity = quantity_tons * (1 + quantity_tolerance)

    candidates = route_data[
        route_data["quantity_tons"].between(
            min_quantity,
            max_quantity
        )
    ].copy()

    if candidates.empty:
        return None

    candidates["quantity_difference"] = (
        abs(candidates["quantity_tons"] - quantity_tons)
        / max(quantity_tons, 1e-6)
    )

    candidates = candidates[
        candidates["quantity_difference"] <= similarity_threshold
    ].copy()

    if candidates.empty:
        return None

    candidates = candidates.sort_values(
        ["quantity_difference", "order_date"],
        ascending=[True, False]
    )

    recent_candidates = candidates.head(10)

    weighted_values = 1 / (
        recent_candidates["quantity_difference"] + 0.05
    )

    estimated_freight = np.average(
        recent_candidates["freight"],
        weights=weighted_values
    )

    return {
        "freight": float(estimated_freight),
        "source": "historical_algorithm",
        "sample_count": int(len(recent_candidates)),
        "lookback_days": lookback_days,
    }
