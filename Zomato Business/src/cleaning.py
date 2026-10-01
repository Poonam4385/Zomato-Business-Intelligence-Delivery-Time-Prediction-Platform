from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .settings import CFG, DATASETS, PROCESSED_DIR, RAW_DIR
from .utils import (
    coerce_numeric,
    get_logger,
    normalize_city,
    parse_mixed_date_series,
    parse_time_series,
    save_df,
    write_json,
)

log = get_logger("cleaning")


@dataclass
class CleanResult:
    tables: Dict[str, pd.DataFrame]
    report: dict


def load_raw() -> Dict[str, pd.DataFrame]:
    tables = {}
    for name, filename in DATASETS.items():
        path = RAW_DIR / filename
        tables[name] = pd.read_csv(path)
        log.info("Loaded %-20s %7d rows", name, len(tables[name]))
    return tables


def _dedupe(df: pd.DataFrame, key: str, report: dict, table: str) -> pd.DataFrame:
    dupes = int(df.duplicated(subset=[key]).sum())
    report[table]["duplicate_key_rows"] = dupes
    return df.drop_duplicates(subset=[key], keep="first").copy()


def _normalize_city_col(df: pd.DataFrame, col: str = "City"):
    if col in df.columns:
        df[col] = df[col].map(normalize_city)
    return df


def _clean_cities(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "cities"
    df = _dedupe(df.copy(), "CityID", report, t)
    _normalize_city_col(df)
    coerce_numeric(df, ["CityID", "Population", "AverageIncome"])
    df = df.dropna(subset=["CityID", "City"])
    df = df.drop_duplicates(subset=["City"], keep="first")
    df["CityID"] = df["CityID"].astype(int)
    df.loc[df["Population"] <= 0, "Population"] = np.nan
    df.loc[df["AverageIncome"] < 0, "AverageIncome"] = np.nan
    return df


def _clean_customers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "customers"
    df = _dedupe(df.copy(), "CustomerID", report, t)
    _normalize_city_col(df)
    coerce_numeric(df, ["CustomerID", "Age", "TotalOrders"])
    invalid_age = (~df["Age"].between(0, 100)) & df["Age"].notna()
    report[t]["invalid_age_rows"] = int(invalid_age.sum())
    df.loc[invalid_age, "Age"] = np.nan
    parsed = parse_mixed_date_series(df["RegistrationDate"], "RegistrationDateClean")
    df = pd.concat([df, parsed], axis=1)
    df["RegistrationDate"] = df["RegistrationDateClean"]
    df = df.drop(columns=["RegistrationDateClean"])
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    df["Membership"] = df["Membership"].astype("string").str.strip()
    df["PreferredCuisine"] = df["PreferredCuisine"].astype("string").str.strip()
    return df


def _clean_restaurants(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "restaurants"
    df = _dedupe(df.copy(), "RestaurantID", report, t)
    _normalize_city_col(df)
    coerce_numeric(df, ["RestaurantID", "Rating", "AverageCost", "Latitude", "Longitude"])
    bad_rating = (~df["Rating"].between(1, 5)) & df["Rating"].notna()
    bad_cost = (df["AverageCost"] < 0) & df["AverageCost"].notna()
    report[t]["invalid_rating_rows"] = int(bad_rating.sum())
    report[t]["negative_average_cost_rows"] = int(bad_cost.sum())
    df.loc[bad_rating, "Rating"] = np.nan
    df.loc[bad_cost, "AverageCost"] = np.nan
    df["OpeningTime"] = parse_time_series(df["OpeningTime"])
    df["ClosingTime"] = parse_time_series(df["ClosingTime"])
    # India-wide bounding box only; this is a coarse sanity flag, not a city validation.
    df["GeoIndiaBoundsFlag"] = (
        df["Latitude"].between(6, 38) & df["Longitude"].between(68, 98)
    ).astype(int)
    return df


def _clean_delivery_partners(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "delivery_partners"
    df = _dedupe(df.copy(), "DeliveryPartnerID", report, t)
    _normalize_city_col(df)
    coerce_numeric(df, ["DeliveryPartnerID", "Age", "Rating", "CompletedDeliveries", "AverageDeliveryTime"])
    bad_age = (~df["Age"].between(16, 70)) & df["Age"].notna()
    bad_rating = (~df["Rating"].between(1, 5)) & df["Rating"].notna()
    df.loc[bad_age, "Age"] = np.nan
    df.loc[bad_rating, "Rating"] = np.nan
    df.loc[df["CompletedDeliveries"] < 0, "CompletedDeliveries"] = np.nan
    df.loc[df["AverageDeliveryTime"] < 0, "AverageDeliveryTime"] = np.nan
    report[t]["invalid_age_rows"] = int(bad_age.sum())
    report[t]["invalid_rating_rows"] = int(bad_rating.sum())
    parsed = parse_mixed_date_series(df["JoiningDate"], "JoiningDateClean")
    df = pd.concat([df, parsed], axis=1)
    df["JoiningDate"] = df["JoiningDateClean"]
    df = df.drop(columns=["JoiningDateClean"])
    return df


def _clean_promotions(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "promotions"
    df = _dedupe(df.copy(), "PromotionID", report, t)
    coerce_numeric(df, ["PromotionID", "DiscountPercentage"])
    s = parse_mixed_date_series(df["StartDate"], "StartDateClean")
    e = parse_mixed_date_series(df["EndDate"], "EndDateClean")
    df = pd.concat([df, s, e], axis=1)
    df["StartDate"], df["EndDate"] = df["StartDateClean"], df["EndDateClean"]
    df = df.drop(columns=["StartDateClean", "EndDateClean"])
    bad_pct = (~df["DiscountPercentage"].between(0, 100)) & df["DiscountPercentage"].notna()
    df.loc[bad_pct, "DiscountPercentage"] = np.nan
    swap = df["EndDate"].notna() & df["StartDate"].notna() & (df["EndDate"] < df["StartDate"])
    report[t]["reversed_date_rows"] = int(swap.sum())
    df["PromoDateSwappedFlag"] = swap.astype(int)
    old_start = df.loc[swap, "StartDate"].copy()
    df.loc[swap, "StartDate"] = df.loc[swap, "EndDate"].values
    df.loc[swap, "EndDate"] = old_start.values
    df["CouponCode"] = df["CouponCode"].astype("string").str.strip()
    return df


def _clean_menu(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "menu"
    df = _dedupe(df.copy(), "FoodItemID", report, t)
    coerce_numeric(df, ["FoodItemID", "RestaurantID", "Price", "PreparationTime", "Calories"])
    bad_price = (df["Price"] < 0) & df["Price"].notna()
    bad_prep = (df["PreparationTime"] < 0) & df["PreparationTime"].notna()
    report[t]["negative_price_rows"] = int(bad_price.sum())
    report[t]["negative_prep_time_rows"] = int(bad_prep.sum())
    df.loc[bad_price, "Price"] = np.nan
    df.loc[bad_prep, "PreparationTime"] = np.nan
    df["Availability"] = df["Availability"].astype("string").str.strip().str.title()
    return df


def _clean_order_items(df: pd.DataFrame, menu: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "order_items"
    df = _dedupe(df.copy(), "OrderItemID", report, t)
    coerce_numeric(df, ["OrderItemID", "OrderID", "FoodItemID", "Quantity", "UnitPrice", "TotalPrice"])
    menu_price = menu[["FoodItemID", "Price"]].rename(columns={"Price": "MenuPrice"})
    df = df.merge(menu_price, on="FoodItemID", how="left")
    bad_qty = (~df["Quantity"].gt(0)) | df["Quantity"].isna()
    bad_unit = (df["UnitPrice"] < 0) | df["UnitPrice"].isna()
    report[t]["invalid_quantity_rows"] = int(bad_qty.sum())
    report[t]["invalid_unit_price_rows"] = int(bad_unit.sum())
    repair_price = bad_unit & df["MenuPrice"].notna() & (df["MenuPrice"] >= 0)
    df.loc[repair_price, "UnitPrice"] = df.loc[repair_price, "MenuPrice"]
    df["UnitPriceRepairedFlag"] = repair_price.astype(int)
    # Quantity cannot be inferred defensibly; drop those rows.
    df = df.loc[~bad_qty].copy()
    df = df.loc[df["UnitPrice"].notna() & (df["UnitPrice"] >= 0)].copy()
    expected = df["Quantity"] * df["UnitPrice"]
    mismatch = df["TotalPrice"].isna() | (~np.isclose(df["TotalPrice"], expected, atol=0.01))
    report[t]["total_price_recomputed_rows"] = int(mismatch.sum())
    df["OriginalTotalPrice"] = df["TotalPrice"]
    df["TotalPrice"] = expected.round(2)
    df["TotalPriceRecomputedFlag"] = mismatch.astype(int)
    df = df.drop(columns=["MenuPrice"])
    return df


def _clean_orders(df: pd.DataFrame, order_items: pd.DataFrame, promotions: pd.DataFrame,
                  customers: pd.DataFrame, restaurants: pd.DataFrame,
                  delivery_partners: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "orders"
    df = _dedupe(df.copy(), "OrderID", report, t)
    coerce_numeric(df, ["OrderID", "CustomerID", "RestaurantID", "DeliveryPartnerID",
                        "DeliveryTimeMinutes", "FoodCost", "DeliveryFee", "Discount", "GST", "FinalAmount"])

    parsed = parse_mixed_date_series(df["OrderDate"], "OrderDateClean")
    df = pd.concat([df, parsed], axis=1)
    df["OrderDate"] = df["OrderDateClean"]
    df = df.rename(columns={
        "OrderDateCleanAmbiguousFlag": "OrderDateAmbiguousFlag",
        "OrderDateCleanParseFailedFlag": "OrderDateParseFailedFlag",
        "OrderDateCleanParseFormat": "OrderDateParseFormat",
    }).drop(columns=["OrderDateClean"])
    df["OrderTime"] = parse_time_series(df["OrderTime"])

    # Referential integrity: remove unrecoverable orphan facts instead of inventing keys.
    valid_customer = set(customers["CustomerID"].dropna().astype(int))
    valid_restaurant = set(restaurants["RestaurantID"].dropna().astype(int))
    valid_partner = set(delivery_partners["DeliveryPartnerID"].dropna().astype(int))
    orphan_c = ~df["CustomerID"].isin(valid_customer)
    orphan_r = ~df["RestaurantID"].isin(valid_restaurant)
    orphan_d = ~df["DeliveryPartnerID"].isin(valid_partner)
    report[t]["orphan_customer_rows"] = int(orphan_c.sum())
    report[t]["orphan_restaurant_rows"] = int(orphan_r.sum())
    report[t]["orphan_partner_rows"] = int(orphan_d.sum())
    df = df.loc[~(orphan_c | orphan_r | orphan_d)].copy()

    # Delivery-time cleaning for analytical use.
    lo = CFG["data"]["delivery_min_minutes"]
    hi = CFG["data"]["delivery_max_minutes"]
    bad_delivery = df["DeliveryTimeMinutes"].notna() & ~df["DeliveryTimeMinutes"].between(lo, hi)
    report[t]["delivery_time_outlier_rows"] = int(bad_delivery.sum())
    df["DeliveryTimeRaw"] = df["DeliveryTimeMinutes"]
    df.loc[bad_delivery, "DeliveryTimeMinutes"] = np.nan

    for col in ["FoodCost", "DeliveryFee", "Discount", "GST", "FinalAmount"]:
        neg = df[col].notna() & (df[col] < 0)
        report[t][f"negative_{col}_rows"] = int(neg.sum())
        df.loc[neg, col] = np.nan

    # Order-item basket total is the authoritative food cost because it is line-item auditable.
    basket = order_items.groupby("OrderID", as_index=False).agg(
        ItemFoodCost=("TotalPrice", "sum"),
        ItemLines=("OrderItemID", "count"),
        TotalQuantity=("Quantity", "sum"),
    )
    df = df.merge(basket, on="OrderID", how="left")
    df["OriginalFoodCost"] = df["FoodCost"]
    item_authoritative = CFG["data"].get("authoritative_food_cost") == "order_items"
    if item_authoritative:
        replace = df["ItemFoodCost"].notna()
        df.loc[replace, "FoodCost"] = df.loc[replace, "ItemFoodCost"]
        df["FoodCostRebuiltFlag"] = replace.astype(int)
    else:
        df["FoodCostRebuiltFlag"] = 0

    # Coupon integrity: invalid codes become null, not fabricated.
    valid_coupon = set(promotions["CouponCode"].dropna().astype(str))
    invalid_coupon = df["CouponCode"].notna() & ~df["CouponCode"].astype(str).isin(valid_coupon)
    report[t]["invalid_coupon_rows"] = int(invalid_coupon.sum())
    df.loc[invalid_coupon, "CouponCode"] = pd.NA

    # Rebuild final amount from auditable components when all components are available.
    df["OriginalFinalAmount"] = df["FinalAmount"]
    components = ["FoodCost", "DeliveryFee", "Discount", "GST"]
    complete = df[components].notna().all(axis=1)
    expected = df["FoodCost"] + df["DeliveryFee"] - df["Discount"] + df["GST"]
    valid_expected = complete & (expected >= 0)
    impossible_expected = complete & (expected < 0)
    mismatch = valid_expected & (df["FinalAmount"].isna() | ~np.isclose(df["FinalAmount"], expected, atol=0.01))
    report[t]["final_amount_recomputed_rows"] = int(mismatch.sum())
    report[t]["financially_impossible_rows"] = int(impossible_expected.sum())
    df.loc[valid_expected, "FinalAmount"] = expected[valid_expected].round(2)
    df.loc[impossible_expected, "FinalAmount"] = np.nan
    df["FinalAmountRebuiltFlag"] = valid_expected.astype(int)

    df["OrderStatus"] = df["OrderStatus"].astype("string").str.strip().str.title()
    # restore canonical capitalization for this dataset
    df["OrderStatus"] = df["OrderStatus"].replace({
        "Food Not Delivered": "Food Not Delivered",
        "Delivered Late": "Delivered Late",
    })
    df["PaymentMethod"] = df["PaymentMethod"].astype("string").str.strip()
    return df


def _clean_payments(df: pd.DataFrame, orders: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "payments"
    df = _dedupe(df.copy(), "PaymentID", report, t)
    coerce_numeric(df, ["PaymentID", "OrderID"])
    parsed = parse_mixed_date_series(df["PaymentDate"], "PaymentDateClean")
    df = pd.concat([df, parsed], axis=1)
    df["PaymentDate"] = df["PaymentDateClean"]
    df = df.drop(columns=["PaymentDateClean"])
    valid = set(orders["OrderID"].dropna().astype(int))
    orphan = ~df["OrderID"].isin(valid)
    report[t]["orphan_order_rows"] = int(orphan.sum())
    return df.loc[~orphan].copy()


def _clean_feedback(df: pd.DataFrame, orders: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "customer_feedback"
    df = _dedupe(df.copy(), "FeedbackID", report, t)
    coerce_numeric(df, ["FeedbackID", "OrderID", "CustomerRating", "DeliveryRating", "FoodRating"])
    for c in ["CustomerRating", "DeliveryRating", "FoodRating"]:
        bad = df[c].notna() & ~df[c].between(1, 5)
        report[t][f"invalid_{c}_rows"] = int(bad.sum())
        df.loc[bad, c] = np.nan
    valid = set(orders["OrderID"].dropna().astype(int))
    orphan = ~df["OrderID"].isin(valid)
    report[t]["orphan_order_rows"] = int(orphan.sum())
    df = df.loc[~orphan].copy()
    df["Sentiment"] = df["Sentiment"].astype("string").str.strip().str.title()
    return df


def _clean_weather(df: pd.DataFrame, cities: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "weather"
    df = _dedupe(df.copy(), "WeatherID", report, t)
    _normalize_city_col(df)
    coerce_numeric(df, ["WeatherID", "Temperature", "Rainfall", "Humidity"])
    parsed = parse_mixed_date_series(df["Date"], "DateClean")
    df = pd.concat([df, parsed], axis=1)
    df["Date"] = df["DateClean"]
    df = df.drop(columns=["DateClean"])
    valid_cities = set(cities["City"].dropna())
    invalid_city = df["City"].isna() | ~df["City"].isin(valid_cities)
    report[t]["missing_or_unknown_city_rows"] = int(invalid_city.sum())
    df = df.loc[~invalid_city].copy()
    df.loc[df["Rainfall"] < 0, "Rainfall"] = np.nan
    df.loc[~df["Humidity"].between(0, 100), "Humidity"] = np.nan
    return df


def _clean_traffic(df: pd.DataFrame, cities: pd.DataFrame, report: dict) -> pd.DataFrame:
    t = "traffic"
    df = _dedupe(df.copy(), "TrafficID", report, t)
    _normalize_city_col(df)
    coerce_numeric(df, ["TrafficID", "AverageSpeed"])
    parsed = parse_mixed_date_series(df["Date"], "DateClean")
    df = pd.concat([df, parsed], axis=1)
    df["Date"] = df["DateClean"]
    df = df.drop(columns=["DateClean"])
    df["Time"] = parse_time_series(df["Time"])
    valid_cities = set(cities["City"].dropna())
    invalid_city = df["City"].isna() | ~df["City"].isin(valid_cities)
    report[t]["missing_or_unknown_city_rows"] = int(invalid_city.sum())
    df = df.loc[~invalid_city].copy()
    df.loc[df["AverageSpeed"] < 0, "AverageSpeed"] = np.nan
    return df


def clean_all(write_outputs: bool = True) -> CleanResult:
    raw = load_raw()
    report = {name: {"raw_rows": int(len(df))} for name, df in raw.items()}

    clean: Dict[str, pd.DataFrame] = {}
    clean["cities"] = _clean_cities(raw["cities"], report)
    clean["customers"] = _clean_customers(raw["customers"], report)
    clean["restaurants"] = _clean_restaurants(raw["restaurants"], report)
    clean["delivery_partners"] = _clean_delivery_partners(raw["delivery_partners"], report)

    # Enforce city dimension integrity before facts are cleaned, so later FK checks remain valid.
    valid_cities = set(clean["cities"]["City"])
    for name in ["customers", "restaurants", "delivery_partners"]:
        bad = clean[name]["City"].isna() | ~clean[name]["City"].isin(valid_cities)
        report[name]["unknown_city_rows"] = int(bad.sum())
        clean[name] = clean[name].loc[~bad].copy()

    clean["promotions"] = _clean_promotions(raw["promotions"], report)
    clean["menu"] = _clean_menu(raw["menu"], report)
    valid_rest = set(clean["restaurants"]["RestaurantID"].dropna().astype(int))
    menu_orphans = ~clean["menu"]["RestaurantID"].isin(valid_rest)
    report["menu"]["orphan_restaurant_rows"] = int(menu_orphans.sum())
    clean["menu"] = clean["menu"].loc[~menu_orphans].copy()

    clean["order_items"] = _clean_order_items(raw["order_items"], clean["menu"], report)
    valid_food = set(clean["menu"]["FoodItemID"].dropna().astype(int))
    food_orphan = ~clean["order_items"]["FoodItemID"].isin(valid_food)
    report["order_items"]["orphan_food_rows"] = int(food_orphan.sum())
    clean["order_items"] = clean["order_items"].loc[~food_orphan].copy()

    clean["orders"] = _clean_orders(raw["orders"], clean["order_items"], clean["promotions"],
                                      clean["customers"], clean["restaurants"], clean["delivery_partners"], report)
    valid_orders = set(clean["orders"]["OrderID"].dropna().astype(int))
    oi_orphan = ~clean["order_items"]["OrderID"].isin(valid_orders)
    report["order_items"]["orphan_order_rows"] = int(oi_orphan.sum())
    clean["order_items"] = clean["order_items"].loc[~oi_orphan].copy()

    clean["payments"] = _clean_payments(raw["payments"], clean["orders"], report)
    clean["customer_feedback"] = _clean_feedback(raw["customer_feedback"], clean["orders"], report)
    clean["weather"] = _clean_weather(raw["weather"], clean["cities"], report)
    clean["traffic"] = _clean_traffic(raw["traffic"], clean["cities"], report)

    # Recompute actual customer order totals from the cleaned fact table.
    actual = clean["orders"].groupby("CustomerID")["OrderID"].nunique().rename("ActualTotalOrders")
    clean["customers"] = clean["customers"].merge(actual, left_on="CustomerID", right_index=True, how="left")
    clean["customers"]["ActualTotalOrders"] = clean["customers"]["ActualTotalOrders"].fillna(0).astype(int)
    mismatch = clean["customers"]["TotalOrders"].fillna(-1).astype(float) != clean["customers"]["ActualTotalOrders"].astype(float)
    report["customers"]["total_orders_mismatch_rows"] = int(mismatch.sum())
    clean["customers"]["OriginalTotalOrders"] = clean["customers"]["TotalOrders"]
    clean["customers"]["TotalOrders"] = clean["customers"]["ActualTotalOrders"]

    for name, df in clean.items():
        report[name]["clean_rows"] = int(len(df))
        report[name]["rows_removed"] = int(report[name]["raw_rows"] - len(df))

    if write_outputs:
        for name, df in clean.items():
            save_df(df, PROCESSED_DIR / f"{name}.pkl")
            save_df(df, PROCESSED_DIR / f"{name}.csv")
        write_json(report, PROCESSED_DIR / "data_quality_report.json")
        pd.DataFrame([
            {"table": table, **metrics} for table, metrics in report.items()
        ]).to_csv(PROCESSED_DIR / "data_quality_summary.csv", index=False)
        log.info("Cleaned datasets written to %s", PROCESSED_DIR)

    return CleanResult(clean, report)


if __name__ == "__main__":
    clean_all(write_outputs=True)
