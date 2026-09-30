import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder


class FreightModel:
    """Gradient-boosted freight prediction model."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None

    def _build_preprocessor(self, X):
        categorical = [
            "origin",
            "destination",
            "route",
            "distance_bucket",
        ]

        numeric = [
            "distance_km",
            "quantity_tons",
            "year",
            "month",
            "weekday",
            "week_of_year",
            "log_quantity",
            "distance_quantity_interaction",
        ]

        self.feature_columns = categorical + numeric

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    categorical,
                ),
                (
                    "numeric",
                    "passthrough",
                    numeric,
                ),
            ],
            remainder="drop",
        )

        return self.preprocessor

    def fit(self, df):
        """Train the model using a chronological train/test split."""

        df = df.sort_values("order_date").reset_index(drop=True)

        split_index = int(len(df) * 0.8)

        train = df.iloc[:split_index]
        test = df.iloc[split_index:]

        X_train = train[self.feature_columns]
        y_train = train["freight"]

        X_test = test[self.feature_columns]
        y_test = test["freight"]

        X_train_encoded = self.preprocessor.fit_transform(X_train)
        X_test_encoded = self.preprocessor.transform(X_test)

        self.model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )

        self.model.fit(X_train_encoded, y_train)

        predictions = self.model.predict(X_test_encoded)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        non_zero = y_test != 0
        mape = (
            np.mean(
                np.abs(
                    (y_test[non_zero] - predictions[non_zero])
                    / y_test[non_zero]
                )
            )
            * 100
        )

        return {
            "train_samples": len(train),
            "test_samples": len(test),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
        }

    def predict(self, df):
        """Generate freight predictions for new records."""

        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        X = df[self.feature_columns]
        X_encoded = self.preprocessor.transform(X)

        return self.model.predict(X_encoded)
