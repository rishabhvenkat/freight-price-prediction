def run_scenario(model, base_features, changes):
    """
    Run a what-if prediction by modifying selected input features.

    Parameters
    ----------
    model:
        Trained FreightModel instance.

    base_features:
        Single-row DataFrame containing model features.

    changes:
        Dictionary of feature names and replacement values.
    """

    scenario = base_features.copy()

    for feature, value in changes.items():
        if feature not in scenario.columns:
            raise ValueError(
                f"Unknown scenario feature: {feature}"
            )

        scenario.loc[:, feature] = value

    prediction = float(model.predict(scenario)[0])

    return {
        "prediction": round(prediction, 2),
        "changes": changes,
    }


def compare_scenarios(model, base_features, scenarios):
    """Compare multiple what-if scenarios against the baseline."""

    baseline = float(model.predict(base_features)[0])

    results = []

    for name, changes in scenarios.items():
        result = run_scenario(
            model=model,
            base_features=base_features,
            changes=changes,
        )

        prediction = result["prediction"]

        results.append({
            "scenario": name,
            "prediction": prediction,
            "delta": round(prediction - baseline, 2),
            "delta_pct": round(
                ((prediction - baseline) / baseline) * 100,
                2,
            ),
            "changes": changes,
        })

    return {
        "baseline": round(baseline, 2),
        "scenarios": results,
    }
