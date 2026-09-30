import numpy as np
import shap


def explain_prediction(model, features, top_n=10):
    """
    Return the most influential transformed features for a prediction.
    """

    if model.model is None:
        raise RuntimeError("Model has not been trained.")

    X = features[model.feature_columns]

    X_encoded = model.preprocessor.transform(X)

    explainer = shap.TreeExplainer(model.model)

    shap_values = explainer.shap_values(X_encoded)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    feature_names = model.preprocessor.get_feature_names_out()

    values = shap_values[0]

    contributions = sorted(
        zip(feature_names, values),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    return [
        {
            "feature": feature,
            "contribution": round(float(value), 4),
            "direction": "positive" if value >= 0 else "negative",
        }
        for feature, value in contributions[:top_n]
    ]
