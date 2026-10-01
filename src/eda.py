from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

from .settings import COMPLETED_STATUSES, FEATURE_DIR, PROCESSED_DIR, REPORT_DIR
from .utils import get_logger

log = get_logger("eda")


def build_eda_report() -> Path:
    orders = pd.read_pickle(PROCESSED_DIR / "orders.pkl")
    restaurants = pd.read_pickle(PROCESSED_DIR / "restaurants.pkl")
    customers = pd.read_pickle(PROCESSED_DIR / "customers.pkl")
    delivery = pd.read_pickle(FEATURE_DIR / "delivery_features.pkl")
    churn = pd.read_pickle(FEATURE_DIR / "churn_features.pkl")

    o = orders.copy()
    o["OrderDate"] = pd.to_datetime(o["OrderDate"], errors="coerce")
    o["Month"] = o["OrderDate"].dt.to_period("M").astype(str)
    completed = o[o["OrderStatus"].isin(COMPLETED_STATUSES)].copy()

    monthly = completed.groupby("Month", as_index=False).agg(
        Orders=("OrderID", "nunique"), Revenue=("FinalAmount", "sum")
    )
    status = o.groupby("OrderStatus", as_index=False)["OrderID"].nunique().rename(columns={"OrderID": "Orders"})
    city = completed.merge(restaurants[["RestaurantID", "City"]], on="RestaurantID", how="left")
    city = city.groupby("City", as_index=False).agg(Orders=("OrderID", "nunique"), Revenue=("FinalAmount", "sum"))
    city = city.sort_values("Revenue", ascending=False).head(15)

    figs = [
        px.line(monthly, x="Month", y="Revenue", markers=True, title="Monthly completed-order revenue"),
        px.bar(status, x="OrderStatus", y="Orders", title="Order status distribution"),
        px.bar(city, x="City", y="Revenue", title="Top restaurant cities by revenue"),
        px.histogram(delivery, x="DeliveryTimeMinutes", nbins=45, title="Clean delivery-time distribution"),
        px.scatter(delivery.sample(min(4000, len(delivery)), random_state=42), x="WeightedPrepTime", y="DeliveryTimeMinutes",
                   trendline=None, title="Delivery time vs weighted preparation time"),
        px.histogram(churn, x="RecencyDays", color="Churn60d", nbins=40, barmode="overlay",
                     title="Recency distribution by 60-day churn label"),
    ]

    kpis = {
        "clean_orders": int(len(o)),
        "completed_orders": int(completed["OrderID"].nunique()),
        "completed_revenue": round(float(completed["FinalAmount"].sum()), 2),
        "customers": int(customers["CustomerID"].nunique()),
        "restaurants": int(restaurants["RestaurantID"].nunique()),
        "avg_delivery_minutes": round(float(delivery["DeliveryTimeMinutes"].mean()), 2),
        "late_rate_pct": round(float((completed["OrderStatus"] == "Delivered Late").mean() * 100), 2),
        "churn_rate_pct": round(float(churn["Churn60d"].mean() * 100), 2),
    }

    html = [
        "<html><head><meta charset='utf-8'><title>Zomato EDA Report</title>",
        "<style>body{font-family:Arial;margin:36px;max-width:1300px} .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpi{padding:14px;border:1px solid #ddd;border-radius:10px}.warn{background:#fff4e5;padding:16px;border-radius:8px}</style>",
        "</head><body>",
        "<h1>Zomato BI & ML — Automated EDA</h1>",
        "<div class='warn'><b>Data caveat:</b> the source is synthetic and contains cross-city entity links and irrecoverably ambiguous source dates. The pipeline flags these instead of fabricating corrections.</div>",
        "<div class='kpis>",
    ]
    # Fix markup while keeping construction simple.
    html[-1] = "<div class='kpis'>"
    for k, v in kpis.items():
        html.append(f"<div class='kpi'><b>{k.replace('_',' ').title()}</b><br><span style='font-size:24px'>{v}</span></div>")
    html.append("</div>")
    for fig in figs:
        html.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
    html.append("</body></html>")

    path = REPORT_DIR / "eda_report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    log.info("EDA report written: %s", path)
    return path


if __name__ == "__main__":
    build_eda_report()
