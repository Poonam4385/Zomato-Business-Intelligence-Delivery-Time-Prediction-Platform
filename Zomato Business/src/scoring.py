from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .settings import FEATURE_DIR, MODEL_DIR


def _sanitize(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    x = df[numeric + categorical].copy()
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
    for c in categorical:
        x[c] = x[c].astype(object)
        x[c] = x[c].where(pd.notna(x[c]), np.nan)
    return x


def score_delivery_order(order_id: int) -> dict:
    bundle = joblib.load(MODEL_DIR / "delivery_model.joblib")
    df = pd.read_pickle(FEATURE_DIR / "delivery_features.pkl")
    row = df[df["OrderID"] == int(order_id)]
    if row.empty:
        raise KeyError(f"OrderID {order_id} not found in delivery feature table")
    x = _sanitize(row, bundle["numeric_features"], bundle["categorical_features"])
    pred = float(bundle["model"].predict(x)[0])
    return {
        "OrderID": int(order_id),
        "actual_delivery_minutes": float(row["DeliveryTimeMinutes"].iloc[0]),
        "predicted_delivery_minutes": round(pred, 2),
        "absolute_error_minutes": round(abs(float(row["DeliveryTimeMinutes"].iloc[0]) - pred), 2),
        "model": bundle["model_name"],
    }


def score_churn_customer(customer_id: int) -> dict:
    bundle = joblib.load(MODEL_DIR / "churn_model.joblib")
    df = pd.read_pickle(FEATURE_DIR / "churn_features.pkl")
    row = df[df["CustomerID"] == int(customer_id)]
    if row.empty:
        raise KeyError(f"CustomerID {customer_id} not found or not eligible for churn scoring")
    x = _sanitize(row, bundle["numeric_features"], bundle["categorical_features"])
    prob = float(bundle["model"].predict_proba(x)[0, 1])
    threshold = float(bundle["threshold"])
    return {
        "CustomerID": int(customer_id),
        "actual_churn_60d": int(row["Churn60d"].iloc[0]),
        "churn_probability": round(prob, 4),
        "threshold": round(threshold, 4),
        "predicted_churn_60d": int(prob >= threshold),
        "model": bundle["model_name"],
    }
