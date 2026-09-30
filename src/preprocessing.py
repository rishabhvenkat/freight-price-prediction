import pandas as pd
import numpy as np


def load_data(path="data/synthetic_freight_data.csv"):
    """Load and validate the synthetic freight dataset."""
    df = pd.read_csv(path)

    required_columns = {
        "order_date",
        "origin",
        "destination",
        "distance_km",
        "quantity_tons",
        "freight",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["order_date"] = pd.to_datetime(df["order_date"])

    return df


def audit_data(df):
    """Return basic data-quality statistics."""
    report = {
        "rows": len(df),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_distance": int((df["distance_km"] <= 0).sum()),
        "invalid_quantity": int((df["quantity_tons"] <= 0).sum()),
        "invalid_freight": int((df["freight"] <= 0).sum()),
    }

    return report


def clean_data(df):
    """Apply generic data-quality rules."""
    df = df.copy()

    df = df.drop_duplicates()

    df = df[
        (df["distance_km"] > 0)
        & (df["quantity_tons"] > 0)
        & (df["freight"] > 0)
    ].copy()

    df["freight_per_ton_km"] = (
        df["freight"] / (df["quantity_tons"] * df["distance_km"])
    )

    # Remove extreme freight-rate outliers using the IQR rule.
    q1 = df["freight_per_ton_km"].quantile(0.25)
    q3 = df["freight_per_ton_km"].quantile(0.75)
    iqr = q3 - q1

    lower = max(0, q1 - 3 * iqr)
    upper = q3 + 3 * iqr

    df = df[
        df["freight_per_ton_km"].between(lower, upper)
    ].copy()

    return df


def create_features(df):
    """Create model features from historical shipment data."""
    df = df.copy()

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["weekday"] = df["order_date"].dt.weekday
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)

    df["route"] = (
        df["origin"].astype(str)
        + "_TO_"
        + df["destination"].astype(str)
    )

    df["distance_bucket"] = pd.cut(
        df["distance_km"],
        bins=[0, 100, 300, 600, 1000, np.inf],
        labels=[
            "0-100",
            "100-300",
            "300-600",
            "600-1000",
            "1000+",
        ],
    ).astype(str)

    df["log_quantity"] = np.log1p(df["quantity_tons"])
    df["distance_quantity_interaction"] = (
        df["distance_km"] * df["quantity_tons"]
    )

    return df
