from src.algorithm import get_algorithm_candidate


def hybrid_predict(
    model,
    historical_data,
    prediction_features,
    origin,
    destination,
    quantity_tons,
    prediction_date,
    algorithm_weight=0.4,
):
    """
    Combine the ML prediction with a historical algorithm candidate.

    If sufficient historical data exists, blend both estimates.
    Otherwise, fall back to the ML prediction.
    """

    ml_prediction = float(model.predict(prediction_features)[0])

    algorithm_result = get_algorithm_candidate(
        historical_data=historical_data,
        origin=origin,
        destination=destination,
        quantity_tons=quantity_tons,
        prediction_date=prediction_date,
    )

    if algorithm_result is None:
        return {
            "freight": ml_prediction,
            "source": "ml",
            "ml_prediction": ml_prediction,
            "algorithm_candidate": None,
        }

    algorithm_prediction = float(algorithm_result["freight"])

    final_prediction = (
        (1 - algorithm_weight) * ml_prediction
        + algorithm_weight * algorithm_prediction
    )

    return {
        "freight": round(final_prediction, 2),
        "source": "hybrid",
        "ml_prediction": round(ml_prediction, 2),
        "algorithm_candidate": round(algorithm_prediction, 2),
        "algorithm_samples": algorithm_result["sample_count"],
    }
