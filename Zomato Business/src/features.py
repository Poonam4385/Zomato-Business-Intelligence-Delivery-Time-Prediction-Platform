from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from .settings import CFG, COMPLETED_STATUSES, FEATURE_DIR, PROCESSED_DIR
from .utils import get_logger, haversine_km, save_df, write_json

log = get_logger("features")


def load_processed() -> Dict[str, pd.DataFrame]:
    names = ["cities", "customers", "restaurants", "menu", "delivery_partners", "promotions",
             "orders", "order_items", "payments", "customer_feedback", "weather", "traffic"]
    return {n: pd.read_pickle(PROCESSED_DIR / f"{n}.pkl") for n in names}


def _combine_date_time(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    date_part = pd.to_datetime(date_s, errors="coerce").dt.strftime("%Y-%m-%d")
    time_part = time_s.astype(str).replace({"NaT": np.nan, "None": np.nan, "<NA>": np.nan})
    return pd.to_datetime(date_part + " " + time_part, errors="coerce")


def _item_features(order_items: pd.DataFrame, menu: pd.DataFrame) -> pd.DataFrame:
    x = order_items.merge(
        menu[["FoodItemID", "PreparationTime", "Category", "Calories"]],
        on="FoodItemID", how="left"
    )
    x["WeightedPrep"] = x["Quantity"] * x["PreparationTime"]
    x["WeightedCalories"] = x["Quantity"] * x["Calories"]
    agg = x.groupby("OrderID", as_index=False).agg(
        ItemLines=("OrderItemID", "count"),
        TotalQuantity=("Quantity", "sum"),
        BasketItemValue=("TotalPrice", "sum"),
        AvgUnitPrice=("UnitPrice", "mean"),
        MaxPrepTime=("PreparationTime", "max"),
        MeanPrepTime=("PreparationTime", "mean"),
        WeightedPrepSum=("WeightedPrep", "sum"),
        WeightedCalories=("WeightedCalories", "sum"),
        MenuCategoryCount=("Category", "nunique"),
    )
    agg["WeightedPrepTime"] = agg["WeightedPrepSum"] / agg["TotalQuantity"].replace(0, np.nan)
    return agg.drop(columns=["WeightedPrepSum"])


def _weather_daily(weather: pd.DataFrame) -> pd.DataFrame:
    # If multiple rows exist for a city/date, aggregate numeric conditions and take the most common label.
    numeric = weather.groupby(["City", "Date"], as_index=False).agg(
        Temperature=("Temperature", "mean"),
        Rainfall=("Rainfall", "mean"),
        Humidity=("Humidity", "mean"),
    )
    mode = (
        weather.dropna(subset=["WeatherCondition"])
        .groupby(["City", "Date"])["WeatherCondition"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    return numeric.merge(mode, on=["City", "Date"], how="left")


def _attach_nearest_traffic(orders: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    tol = pd.Timedelta(minutes=int(CFG["data"]["traffic_match_tolerance_minutes"]))
    t = traffic.copy()
    t["TrafficTimestamp"] = _combine_date_time(t["Date"], t["Time"])
    t = t.dropna(subset=["City", "TrafficTimestamp"])
    pieces = []
    for city, left in orders.groupby("RestaurantCity", dropna=False):
        left = left.copy().sort_values("OrderTimestamp")
        if pd.isna(city):
            left["TrafficLevel"] = pd.NA
            left["AverageSpeed"] = np.nan
            left["TrafficTimeDeltaMinutes"] = np.nan
            pieces.append(left)
            continue
        right = t[t["City"] == city][["TrafficTimestamp", "TrafficLevel", "AverageSpeed"]].sort_values("TrafficTimestamp")
        if right.empty:
            left["TrafficLevel"] = pd.NA
            left["AverageSpeed"] = np.nan
            left["TrafficTimeDeltaMinutes"] = np.nan
            pieces.append(left)
            continue
        m = pd.merge_asof(
            left, right,
            left_on="OrderTimestamp", right_on="TrafficTimestamp",
            direction="nearest", tolerance=tol
        )
        m["TrafficTimeDeltaMinutes"] = (
            (m["OrderTimestamp"] - m["TrafficTimestamp"]).abs().dt.total_seconds() / 60
        )
        pieces.append(m)
    return pd.concat(pieces, ignore_index=True).sort_values("OrderTimestamp")


def build_delivery_features(write_output: bool = True) -> pd.DataFrame:
    d = load_processed()
    o = d["orders"].copy()
    o["OrderTimestamp"] = _combine_date_time(o["OrderDate"], o["OrderTime"])

    # Keep delivered outcomes for the regression target. Status itself is never used as a feature.
    o = o[o["OrderStatus"].isin(COMPLETED_STATUSES)].copy()
    o = o.dropna(subset=["OrderTimestamp", "DeliveryTimeMinutes"])

    c = d["customers"][["CustomerID", "Age", "Gender", "City", "Membership", "PreferredCuisine", "RegistrationDate"]].rename(
        columns={"City": "CustomerCity", "Age": "CustomerAge", "Gender": "CustomerGender"}
    )
    r = d["restaurants"][["RestaurantID", "Cuisine", "City", "Area", "Rating", "AverageCost", "RestaurantType", "Latitude", "Longitude"]].rename(
        columns={
            "Cuisine": "RestaurantCuisine", "City": "RestaurantCity", "Rating": "RestaurantRating",
            "AverageCost": "RestaurantAverageCost", "Latitude": "RestaurantLatitude", "Longitude": "RestaurantLongitude"
        }
    )
    dp = d["delivery_partners"][["DeliveryPartnerID", "Age", "Gender", "VehicleType", "JoiningDate", "City", "Rating", "CompletedDeliveries", "AverageDeliveryTime"]].rename(
        columns={
            "Age": "PartnerAge", "Gender": "PartnerGender", "City": "PartnerCity", "Rating": "PartnerRating",
            "AverageDeliveryTime": "PartnerHistoricalAvgDelivery"
        }
    )
    promo = d["promotions"][["CouponCode", "DiscountPercentage", "CampaignName"]].drop_duplicates("CouponCode")

    x = (o.merge(c, on="CustomerID", how="left")
           .merge(r, on="RestaurantID", how="left")
           .merge(dp, on="DeliveryPartnerID", how="left")
           .merge(promo, on="CouponCode", how="left")
           .merge(_item_features(d["order_items"], d["menu"]), on="OrderID", how="left"))

    x["OrderHour"] = x["OrderTimestamp"].dt.hour
    x["OrderMinute"] = x["OrderTimestamp"].dt.minute
    x["DayOfWeek"] = x["OrderTimestamp"].dt.day_name()
    x["Month"] = x["OrderTimestamp"].dt.month
    x["IsWeekend"] = x["OrderTimestamp"].dt.dayofweek.isin([5, 6]).astype(int)
    x["IsLunch"] = x["OrderHour"].between(11, 14).astype(int)
    x["IsDinner"] = x["OrderHour"].between(18, 22).astype(int)
    x["CustomerTenureDays"] = (x["OrderTimestamp"].dt.normalize() - pd.to_datetime(x["RegistrationDate"])).dt.days.clip(lower=0)
    x["PartnerTenureDays"] = (x["OrderTimestamp"].dt.normalize() - pd.to_datetime(x["JoiningDate"])).dt.days.clip(lower=0)
    x["DiscountPctOfFood"] = 100 * x["Discount"] / x["FoodCost"].replace(0, np.nan)
    x["DeliveryFeePctOfFood"] = 100 * x["DeliveryFee"] / x["FoodCost"].replace(0, np.nan)
    x["HasCoupon"] = x["CouponCode"].notna().astype(int)
    x["CustomerRestaurantCityMismatch"] = (x["CustomerCity"] != x["RestaurantCity"]).astype(int)
    x["RestaurantPartnerCityMismatch"] = (x["RestaurantCity"] != x["PartnerCity"]).astype(int)

    # Optional distance feature: automatically activates if future data includes delivery coordinates.
    if {"DeliveryLatitude", "DeliveryLongitude"}.issubset(x.columns):
        x["DistanceKm"] = haversine_km(
            x["RestaurantLatitude"], x["RestaurantLongitude"], x["DeliveryLatitude"], x["DeliveryLongitude"]
        )
        x["DistanceFeatureAvailable"] = 1
    else:
        x["DistanceKm"] = np.nan
        x["DistanceFeatureAvailable"] = 0

    # Weather matched to restaurant city/date; restaurant city is the operational location of the kitchen.
    w = _weather_daily(d["weather"]).rename(columns={"City": "RestaurantCity", "Date": "OrderDate"})
    x = x.merge(w, on=["RestaurantCity", "OrderDate"], how="left")
    x = _attach_nearest_traffic(x, d["traffic"])

    # Leakage-safe historical behavior: shift before expanding so the current target never enters its own features.
    x = x.sort_values("OrderTimestamp").reset_index(drop=True)
    for key, prefix in [
        ("CustomerID", "Customer"), ("RestaurantID", "Restaurant"), ("DeliveryPartnerID", "Partner")
    ]:
        x[f"{prefix}PriorOrderCount"] = x.groupby(key).cumcount()
        x[f"{prefix}PriorAvgDelivery"] = x.groupby(key)["DeliveryTimeMinutes"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

    x["CustomerPriorSpend"] = x.groupby("CustomerID")["FinalAmount"].transform(
        lambda s: s.shift(1).fillna(0).cumsum()
    )

    if CFG["data"].get("strict_city_match_for_model", False):
        x = x[(x["CustomerRestaurantCityMismatch"] == 0) & (x["RestaurantPartnerCityMismatch"] == 0)].copy()

    if write_output:
        save_df(x, FEATURE_DIR / "delivery_features.pkl")
        save_df(x, FEATURE_DIR / "delivery_features.csv")
        write_json({
            "rows": len(x),
            "target_non_null": int(x["DeliveryTimeMinutes"].notna().sum()),
            "cross_city_customer_restaurant_pct": round(float(x["CustomerRestaurantCityMismatch"].mean() * 100), 2),
            "cross_city_restaurant_partner_pct": round(float(x["RestaurantPartnerCityMismatch"].mean() * 100), 2),
            "distance_feature_available": bool(x["DistanceFeatureAvailable"].max() == 1),
        }, FEATURE_DIR / "delivery_feature_manifest.json")
        log.info("Delivery feature table: %d rows, %d columns", *x.shape)
    return x


def build_churn_features(write_output: bool = True) -> pd.DataFrame:
    d = load_processed()
    o = d["orders"].copy()
    o["OrderDate"] = pd.to_datetime(o["OrderDate"], errors="coerce")
    max_date = o["OrderDate"].max()
    horizon_days = int(CFG["modeling"]["churn_horizon_days"])
    snapshot = max_date - pd.Timedelta(days=horizon_days)
    outcome_end = snapshot + pd.Timedelta(days=horizon_days)

    hist = o[o["OrderDate"] <= snapshot].copy()
    future = o[(o["OrderDate"] > snapshot) & (o["OrderDate"] <= outcome_end)].copy()
    completed_hist = hist[hist["OrderStatus"].isin(COMPLETED_STATUSES)].copy()

    # One row per customer with historical behavior only.
    base = d["customers"][["CustomerID", "Age", "Gender", "City", "RegistrationDate", "Membership", "PreferredCuisine"]].copy()
    base = base[pd.to_datetime(base["RegistrationDate"], errors="coerce") <= snapshot].copy()

    last_order = hist.groupby("CustomerID")["OrderDate"].max().rename("LastOrderDate")
    order_count = hist.groupby("CustomerID")["OrderID"].nunique().rename("HistoricalOrders")
    completed_count = completed_hist.groupby("CustomerID")["OrderID"].nunique().rename("CompletedOrders")
    monetary = completed_hist.groupby("CustomerID")["FinalAmount"].sum().rename("HistoricalSpend")
    avg_basket = completed_hist.groupby("CustomerID")["FinalAmount"].mean().rename("AvgOrderValue")
    avg_delivery = completed_hist.groupby("CustomerID")["DeliveryTimeMinutes"].mean().rename("AvgDeliveryMinutes")
    cancellation = hist.assign(IsCancelled=(hist["OrderStatus"] == "Cancelled").astype(int)).groupby("CustomerID")["IsCancelled"].mean().rename("CancellationRate")
    late = hist.assign(IsLate=(hist["OrderStatus"] == "Delivered Late").astype(int)).groupby("CustomerID")["IsLate"].mean().rename("LateRate")
    restaurants = hist.groupby("CustomerID")["RestaurantID"].nunique().rename("RestaurantVariety")
    payment_variety = hist.groupby("CustomerID")["PaymentMethod"].nunique().rename("PaymentMethodVariety")
    discount_use = hist.groupby("CustomerID")["Discount"].mean().rename("AvgDiscount")

    feats = base.set_index("CustomerID").join([
        last_order, order_count, completed_count, monetary, avg_basket, avg_delivery,
        cancellation, late, restaurants, payment_variety, discount_use
    ]).reset_index()

    feats["RecencyDays"] = (snapshot - pd.to_datetime(feats["LastOrderDate"])).dt.days
    feats["CustomerTenureDays"] = (snapshot - pd.to_datetime(feats["RegistrationDate"])).dt.days.clip(lower=0)
    feats["OrdersPer30Days"] = 30 * feats["HistoricalOrders"] / feats["CustomerTenureDays"].replace(0, np.nan)
    feats["SpendPerOrder"] = feats["HistoricalSpend"] / feats["CompletedOrders"].replace(0, np.nan)
    feats["HasOrderedHistorically"] = feats["HistoricalOrders"].fillna(0).gt(0).astype(int)

    future_orderers = set(future["CustomerID"].dropna().astype(int))
    feats["Churn60d"] = (~feats["CustomerID"].astype(int).isin(future_orderers)).astype(int)

    # Churn is meaningful only after at least one historical order; preserve cold-start users separately.
    feats["EligibleForChurnModel"] = feats["HistoricalOrders"].fillna(0).ge(1).astype(int)
    model_feats = feats[feats["EligibleForChurnModel"] == 1].copy()

    if write_output:
        save_df(feats, FEATURE_DIR / "churn_features_all_customers.pkl")
        save_df(model_feats, FEATURE_DIR / "churn_features.pkl")
        save_df(model_feats, FEATURE_DIR / "churn_features.csv")
        write_json({
            "snapshot_date": str(snapshot.date()),
            "outcome_end_date": str(outcome_end.date()),
            "horizon_days": horizon_days,
            "eligible_customers": len(model_feats),
            "churn_rate": round(float(model_feats["Churn60d"].mean()), 4) if len(model_feats) else None,
        }, FEATURE_DIR / "churn_feature_manifest.json")
        log.info("Churn feature table: %d eligible customers", len(model_feats))
    return model_feats


def build_all_features():
    return build_delivery_features(True), build_churn_features(True)


if __name__ == "__main__":
    build_all_features()
