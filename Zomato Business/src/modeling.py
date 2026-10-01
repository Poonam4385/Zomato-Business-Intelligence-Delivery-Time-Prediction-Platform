from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .settings import CFG, FEATURE_DIR, MODEL_DIR, RANDOM_STATE, REPORT_DIR
from .utils import get_logger, write_json

log = get_logger("modeling")


def _regression_features(df: pd.DataFrame):
    numeric = [
        "FoodCost", "DeliveryFee", "Discount", "GST", "FinalAmount", "OrderHour", "OrderMinute",
        "Month", "IsWeekend", "IsLunch", "IsDinner", "CustomerAge", "RestaurantRating",
        "RestaurantAverageCost", "PartnerAge", "PartnerRating", "CompletedDeliveries",
        "PartnerHistoricalAvgDelivery", "DiscountPercentage", "ItemLines", "TotalQuantity",
        "BasketItemValue", "AvgUnitPrice", "MaxPrepTime", "MeanPrepTime", "WeightedPrepTime",
        "WeightedCalories", "MenuCategoryCount", "CustomerTenureDays", "PartnerTenureDays",
        "DiscountPctOfFood", "DeliveryFeePctOfFood", "HasCoupon", "Temperature", "Rainfall", "Humidity",
        "AverageSpeed", "TrafficTimeDeltaMinutes", "CustomerPriorOrderCount", "CustomerPriorAvgDelivery",
        "RestaurantPriorOrderCount", "RestaurantPriorAvgDelivery", "PartnerPriorOrderCount",
        "PartnerPriorAvgDelivery", "CustomerPriorSpend", "CustomerRestaurantCityMismatch",
        "RestaurantPartnerCityMismatch", "DistanceKm", "DistanceFeatureAvailable",
    ]
    categorical = [
        "PaymentMethod", "Membership", "PreferredCuisine", "RestaurantCuisine", "RestaurantCity",
        "Area", "RestaurantType", "VehicleType", "DayOfWeek", "WeatherCondition", "TrafficLevel",
        "CampaignName",
    ]
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def _classification_features(df: pd.DataFrame):
    numeric = [
        "Age", "HistoricalOrders", "CompletedOrders", "HistoricalSpend", "AvgOrderValue",
        "AvgDeliveryMinutes", "CancellationRate", "LateRate", "RestaurantVariety",
        "PaymentMethodVariety", "AvgDiscount", "RecencyDays", "CustomerTenureDays",
        "OrdersPer30Days", "SpendPerOrder", "HasOrderedHistorically",
    ]
    categorical = ["Gender", "City", "Membership", "PreferredCuisine"]
    return [c for c in numeric if c in df.columns], [c for c in categorical if c in df.columns]


def _preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool = False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=int(CFG["modeling"]["min_category_frequency"]),
            sparse_output=False,
        )),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)


def _sanitize_X(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    x = df[numeric + categorical].copy()
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
    for c in categorical:
        x[c] = x[c].astype(object)
        x[c] = x[c].where(pd.notna(x[c]), np.nan)
    return x


def _reg_metrics(y_true, pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, pred)), 4),
        "RMSE": round(float(mean_squared_error(y_true, pred) ** 0.5), 4),
        "R2": round(float(r2_score(y_true, pred)), 4),
    }


def train_delivery_model() -> dict[str, Any]:
    df = pd.read_pickle(FEATURE_DIR / "delivery_features.pkl").sort_values("OrderTimestamp").copy()
    target = "DeliveryTimeMinutes"
    numeric, categorical = _regression_features(df)
    numeric = [c for c in numeric if df[c].notna().any()]
    categorical = [c for c in categorical if df[c].notna().any()]
    features = numeric + categorical
    model_df = df.dropna(subset=[target, "OrderTimestamp"]).copy()

    # Time-based split prevents future orders from informing the past.
    split_idx = int(len(model_df) * (1 - float(CFG["modeling"]["delivery_test_fraction"])))
    train = model_df.iloc[:split_idx].copy()
    test = model_df.iloc[split_idx:].copy()
    X_train, y_train = _sanitize_X(train, numeric, categorical), train[target]
    X_test, y_test = _sanitize_X(test, numeric, categorical), test[target]

    baseline = np.full(len(y_test), y_train.mean())
    results = {"baseline_mean": _reg_metrics(y_test, baseline)}

    candidates = {
        "ridge": Pipeline([
            ("prep", _preprocessor(numeric, categorical, scale_numeric=True)),
            ("model", Ridge(alpha=5.0)),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor(numeric, categorical)),
            ("model", RandomForestRegressor(
                n_estimators=int(CFG["modeling"]["n_estimators"]),
                max_depth=int(CFG["modeling"]["max_depth"]),
                min_samples_leaf=int(CFG["modeling"]["min_samples_leaf"]),
                n_jobs=int(CFG["modeling"]["n_jobs"]), random_state=RANDOM_STATE,
            )),
        ]),
        "extra_trees": Pipeline([
            ("prep", _preprocessor(numeric, categorical)),
            ("model", ExtraTreesRegressor(
                n_estimators=int(CFG["modeling"]["n_estimators"]),
                max_depth=int(CFG["modeling"]["max_depth"]),
                min_samples_leaf=int(CFG["modeling"]["min_samples_leaf"]),
                n_jobs=int(CFG["modeling"]["n_jobs"]), random_state=RANDOM_STATE,
            )),
        ]),
    }

    fitted = {}
    for name, pipe in candidates.items():
        log.info("Training delivery model: %s", name)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = _reg_metrics(y_test, pred)
        fitted[name] = pipe

    best_name = min(candidates.keys(), key=lambda n: results[n]["MAE"])
    best = fitted[best_name]
    best_pred = best.predict(X_test)
    joblib.dump({
        "model": best, "features": features, "numeric_features": numeric,
        "categorical_features": categorical, "target": target,
        "model_name": best_name, "metrics": results[best_name],
    }, MODEL_DIR / "delivery_model.joblib")

    pred_df = test[["OrderID", "OrderTimestamp", target]].copy()
    pred_df["PredictedDeliveryTimeMinutes"] = best_pred
    pred_df["AbsoluteError"] = (pred_df[target] - pred_df["PredictedDeliveryTimeMinutes"]).abs()
    pred_df.to_csv(REPORT_DIR / "delivery_test_predictions.csv", index=False)

    report = {
        "split": {
            "train_rows": len(train), "test_rows": len(test),
            "train_end": str(train["OrderTimestamp"].max()),
            "test_start": str(test["OrderTimestamp"].min()),
        },
        "features": features,
        "models": results,
        "best_model": best_name,
        "best_metrics": results[best_name],
        "beats_mean_baseline": results[best_name]["MAE"] < results["baseline_mean"]["MAE"],
        "prd_target_comment": "The source target has weak observable signal. Treat MAE<5/RMSE<6 as a dataset-quality hypothesis, not a guaranteed model KPI.",
    }
    write_json(report, REPORT_DIR / "delivery_model_report.json")
    return report


def _class_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y_true, prob)), 4) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, zero_division=0, output_dict=True),
    }


def train_churn_model() -> dict[str, Any]:
    df = pd.read_pickle(FEATURE_DIR / "churn_features.pkl").copy()
    target = "Churn60d"
    numeric, categorical = _classification_features(df)
    numeric = [c for c in numeric if df[c].notna().any()]
    categorical = [c for c in categorical if df[c].notna().any()]
    features = numeric + categorical
    X, y = _sanitize_X(df, numeric, categorical), df[target].astype(int)

    test_frac = float(CFG["modeling"]["churn_test_fraction"])
    val_frac = float(CFG["modeling"]["churn_validation_fraction"])
    X_trainval, X_test, y_trainval, y_test, id_trainval, id_test = train_test_split(
        X, y, df["CustomerID"], test_size=test_frac, stratify=y, random_state=RANDOM_STATE
    )
    val_relative = val_frac / (1 - test_frac)
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_trainval, y_trainval, id_trainval, test_size=val_relative,
        stratify=y_trainval, random_state=RANDOM_STATE
    )

    candidates = {
        "logistic": Pipeline([
            ("prep", _preprocessor(numeric, categorical, scale_numeric=True)),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor(numeric, categorical)),
            ("model", RandomForestClassifier(
                n_estimators=int(CFG["modeling"]["n_estimators"]),
                max_depth=int(CFG["modeling"]["max_depth"]),
                min_samples_leaf=int(CFG["modeling"]["min_samples_leaf"]),
                class_weight="balanced", n_jobs=int(CFG["modeling"]["n_jobs"]), random_state=RANDOM_STATE,
            )),
        ]),
        "extra_trees": Pipeline([
            ("prep", _preprocessor(numeric, categorical)),
            ("model", ExtraTreesClassifier(
                n_estimators=int(CFG["modeling"]["n_estimators"]),
                max_depth=int(CFG["modeling"]["max_depth"]),
                min_samples_leaf=int(CFG["modeling"]["min_samples_leaf"]),
                class_weight="balanced", n_jobs=int(CFG["modeling"]["n_jobs"]), random_state=RANDOM_STATE,
            )),
        ]),
    }

    leaderboard = {}
    fitted = {}
    tuned_threshold = {}
    for name, pipe in candidates.items():
        log.info("Training churn model: %s", name)
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.10, 0.90, 81)
        f1s = [f1_score(y_val, val_prob >= t, zero_division=0) for t in thresholds]
        t_best = float(thresholds[int(np.argmax(f1s))])
        leaderboard[name] = _class_metrics(y_val, val_prob, t_best)
        tuned_threshold[name] = t_best
        fitted[name] = pipe

    best_name = max(candidates.keys(), key=lambda n: leaderboard[n]["pr_auc"] or -1)
    best = fitted[best_name]
    threshold = tuned_threshold[best_name]
    test_prob = best.predict_proba(X_test)[:, 1]
    test_metrics = _class_metrics(y_test, test_prob, threshold)

    joblib.dump({
        "model": best, "features": features, "numeric_features": numeric,
        "categorical_features": categorical, "target": target,
        "model_name": best_name, "threshold": threshold, "metrics": test_metrics,
    }, MODEL_DIR / "churn_model.joblib")

    pred_df = pd.DataFrame({
        "CustomerID": id_test.values,
        "ActualChurn60d": y_test.values,
        "ChurnProbability": test_prob,
        "PredictedChurn60d": (test_prob >= threshold).astype(int),
    }).sort_values("ChurnProbability", ascending=False)
    pred_df.to_csv(REPORT_DIR / "churn_test_predictions.csv", index=False)

    report = {
        "rows": len(df), "churn_rate": round(float(y.mean()), 4),
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "validation_leaderboard": leaderboard,
        "best_model": best_name,
        "selected_threshold": threshold,
        "test_metrics": test_metrics,
        "features": features,
    }
    write_json(report, REPORT_DIR / "churn_model_report.json")
    return report


def train_all_models():
    return train_delivery_model(), train_churn_model()


if __name__ == "__main__":
    print(json.dumps(train_all_models(), indent=2, default=str))
