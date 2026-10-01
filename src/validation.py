from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .settings import COMPLETED_STATUSES, PROCESSED_DIR, REPORT_DIR
from .utils import get_logger, write_json

log = get_logger("validation")


def _load(name: str) -> pd.DataFrame:
    return pd.read_pickle(PROCESSED_DIR / f"{name}.pkl")


def validate_processed() -> dict:
    names = ["cities", "customers", "restaurants", "menu", "delivery_partners", "promotions",
             "orders", "order_items", "payments", "customer_feedback", "weather", "traffic"]
    d = {n: _load(n) for n in names}
    out = {"checks": {}, "warnings": []}

    def check(name: str, ok: bool, detail: str):
        out["checks"][name] = {"passed": bool(ok), "detail": detail}

    # Primary key uniqueness.
    keys = {
        "cities": "CityID", "customers": "CustomerID", "restaurants": "RestaurantID",
        "menu": "FoodItemID", "delivery_partners": "DeliveryPartnerID", "promotions": "PromotionID",
        "orders": "OrderID", "order_items": "OrderItemID", "payments": "PaymentID",
        "customer_feedback": "FeedbackID", "weather": "WeatherID", "traffic": "TrafficID",
    }
    for n, k in keys.items():
        dup = int(d[n].duplicated(k).sum())
        check(f"{n}.pk_unique", dup == 0, f"duplicate {k} rows={dup}")

    # Key referential checks.
    fk_specs = [
        ("orders", "CustomerID", "customers", "CustomerID"),
        ("orders", "RestaurantID", "restaurants", "RestaurantID"),
        ("orders", "DeliveryPartnerID", "delivery_partners", "DeliveryPartnerID"),
        ("menu", "RestaurantID", "restaurants", "RestaurantID"),
        ("order_items", "OrderID", "orders", "OrderID"),
        ("order_items", "FoodItemID", "menu", "FoodItemID"),
        ("payments", "OrderID", "orders", "OrderID"),
        ("customer_feedback", "OrderID", "orders", "OrderID"),
    ]
    for child, fk, parent, pk in fk_specs:
        bad = ~d[child][fk].isin(set(d[parent][pk]))
        count = int(bad.sum())
        check(f"fk.{child}.{fk}", count == 0, f"orphan rows={count}")

    # Financial equation.
    o = d["orders"].copy()
    complete = o[["FoodCost", "DeliveryFee", "Discount", "GST", "FinalAmount"]].notna().all(axis=1)
    expected = o["FoodCost"] + o["DeliveryFee"] - o["Discount"] + o["GST"]
    mismatch = complete & ~np.isclose(expected, o["FinalAmount"], atol=0.01)
    check("orders.financial_equation", int(mismatch.sum()) == 0, f"mismatched rows={int(mismatch.sum())}")

    # Structural city consistency is a warning because source data cannot be safely invented.
    x = o[["OrderID", "CustomerID", "RestaurantID", "DeliveryPartnerID"]].merge(
        d["customers"][["CustomerID", "City"]].rename(columns={"City": "CustomerCity"}), on="CustomerID", how="left"
    ).merge(
        d["restaurants"][["RestaurantID", "City"]].rename(columns={"City": "RestaurantCity"}), on="RestaurantID", how="left"
    ).merge(
        d["delivery_partners"][["DeliveryPartnerID", "City"]].rename(columns={"City": "PartnerCity"}), on="DeliveryPartnerID", how="left"
    )
    cross = (x["CustomerCity"] != x["RestaurantCity"]) | (x["RestaurantCity"] != x["PartnerCity"])
    pct = float(cross.mean() * 100)
    out["warnings"].append({
        "name": "cross_city_order_relationships",
        "value_pct": round(pct, 2),
        "message": "Synthetic source links many orders across different operational cities. Keep as an audit flag; do not fabricate corrected entity links."
    })

    # Ambiguous order dates.
    amb = int(o.get("OrderDateAmbiguousFlag", pd.Series(dtype=int)).fillna(False).sum())
    out["warnings"].append({
        "name": "ambiguous_order_dates",
        "count": amb,
        "message": "Numeric source dates with both components <=12 are inherently ambiguous; configured parsing policy was applied and flags retained."
    })

    completed = o[o["OrderStatus"].isin(COMPLETED_STATUSES)]
    out["summary"] = {
        "orders": int(len(o)),
        "completed_orders": int(len(completed)),
        "completed_revenue": round(float(completed["FinalAmount"].sum()), 2),
        "date_min": str(o["OrderDate"].min().date()) if o["OrderDate"].notna().any() else None,
        "date_max": str(o["OrderDate"].max().date()) if o["OrderDate"].notna().any() else None,
    }

    passed = sum(1 for v in out["checks"].values() if v["passed"])
    out["test_summary"] = {"passed": passed, "total": len(out["checks"]), "all_passed": passed == len(out["checks"])}
    write_json(out, REPORT_DIR / "validation_report.json")
    log.info("Validation: %d/%d hard checks passed", passed, len(out["checks"]))
    return out


if __name__ == "__main__":
    print(json.dumps(validate_processed(), indent=2))
