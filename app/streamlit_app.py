from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring import score_churn_customer, score_delivery_order
from src.settings import COMPLETED_STATUSES, FEATURE_DIR, PROCESSED_DIR, REPORT_DIR

st.set_page_config(page_title="Zomato BI & ML", layout="wide")
st.title("Zomato Business Intelligence & ML Platform")
st.caption("Cleaned analytics, data-quality audit, delivery-time regression and temporal 60-day churn modeling")


def read_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_data():
    orders = pd.read_pickle(PROCESSED_DIR / "orders.pkl")
    restaurants = pd.read_pickle(PROCESSED_DIR / "restaurants.pkl")
    delivery = pd.read_pickle(FEATURE_DIR / "delivery_features.pkl")
    churn = pd.read_pickle(FEATURE_DIR / "churn_features.pkl")
    return orders, restaurants, delivery, churn


try:
    orders, restaurants, delivery, churn = load_data()
except FileNotFoundError:
    st.error("Processed data not found. Run: python run_pipeline.py")
    st.stop()

orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], errors="coerce")
completed = orders[orders["OrderStatus"].isin(COMPLETED_STATUSES)].copy()

with st.sidebar:
    st.header("Filters")
    min_date = orders["OrderDate"].min().date()
    max_date = orders["OrderDate"].max().date()
    date_range = st.date_input("Order date", (min_date, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask = orders["OrderDate"].between(start, end)
        filtered_orders = orders[mask].copy()
    else:
        filtered_orders = orders.copy()

pages = st.tabs(["Executive BI", "Data Quality", "Delivery ML", "Churn ML", "Scoring Demo"])

with pages[0]:
    comp = filtered_orders[filtered_orders["OrderStatus"].isin(COMPLETED_STATUSES)]
    late_rate = (comp["OrderStatus"] == "Delivered Late").mean() * 100 if len(comp) else 0
    cancel_rate = (filtered_orders["OrderStatus"] == "Cancelled").mean() * 100 if len(filtered_orders) else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orders", f"{len(filtered_orders):,}")
    c2.metric("Completed", f"{len(comp):,}")
    c3.metric("Completed revenue", f"₹{comp['FinalAmount'].sum():,.0f}")
    c4.metric("Late rate", f"{late_rate:.1f}%")
    c5.metric("Cancellation rate", f"{cancel_rate:.1f}%")

    monthly = comp.assign(Month=comp["OrderDate"].dt.to_period("M").astype(str)).groupby("Month", as_index=False).agg(
        Revenue=("FinalAmount", "sum"), Orders=("OrderID", "nunique")
    )
    st.plotly_chart(px.line(monthly, x="Month", y="Revenue", markers=True, title="Monthly completed-order revenue"), use_container_width=True)

    city = comp.merge(restaurants[["RestaurantID", "City"]], on="RestaurantID", how="left").groupby("City", as_index=False).agg(
        Revenue=("FinalAmount", "sum"), Orders=("OrderID", "nunique")
    ).sort_values("Revenue", ascending=False)
    st.plotly_chart(px.bar(city.head(15), x="City", y="Revenue", title="Top operational restaurant cities"), use_container_width=True)

    s1, s2 = st.columns(2)
    with s1:
        status = filtered_orders["OrderStatus"].value_counts().rename_axis("OrderStatus").reset_index(name="Orders")
        st.plotly_chart(px.pie(status, names="OrderStatus", values="Orders", title="Order status mix"), use_container_width=True)
    with s2:
        st.plotly_chart(px.histogram(delivery, x="DeliveryTimeMinutes", nbins=40, title="Delivery-time distribution"), use_container_width=True)

with pages[1]:
    val = read_json(REPORT_DIR / "validation_report.json")
    dq = read_json(PROCESSED_DIR / "data_quality_report.json")
    if val:
        st.success(f"Hard validation checks passed: {val['test_summary']['passed']}/{val['test_summary']['total']}")
        st.subheader("Known source-data warnings")
        for w in val.get("warnings", []):
            st.warning(f"{w['name']}: {w.get('value_pct', w.get('count', ''))} — {w['message']}")
    if dq:
        rows = []
        for table, metrics in dq.items():
            rows.append({"table": table, **metrics})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with pages[2]:
    report = read_json(REPORT_DIR / "delivery_model_report.json")
    st.info("The supplied synthetic target has weak predictive signal. This page reports that limitation rather than hiding it.")
    if report:
        st.json({"best_model": report.get("best_model"), "best_metrics": report.get("best_metrics"), "baseline": report.get("models", {}).get("baseline_mean")})
    sample = delivery.sample(min(4000, len(delivery)), random_state=42)
    st.plotly_chart(px.scatter(sample, x="WeightedPrepTime", y="DeliveryTimeMinutes", color="TrafficLevel",
                               title="Delivery time vs preparation time, colored by traffic"), use_container_width=True)

with pages[3]:
    report = read_json(REPORT_DIR / "churn_model_report.json")
    if report:
        st.json({"best_model": report.get("best_model"), "test_metrics": report.get("test_metrics")})
    st.plotly_chart(px.histogram(churn, x="RecencyDays", color="Churn60d", barmode="overlay", nbins=40,
                                 title="Recency by 60-day churn label"), use_container_width=True)
    top = churn[["CustomerID", "HistoricalOrders", "HistoricalSpend", "RecencyDays", "Churn60d"]].sort_values("RecencyDays", ascending=False).head(100)
    st.dataframe(top, use_container_width=True, hide_index=True)

with pages[4]:
    st.subheader("Score an existing engineered record")
    a, b = st.columns(2)
    with a:
        oid = st.selectbox("OrderID", delivery["OrderID"].head(500).astype(int).tolist())
        if st.button("Predict delivery time"):
            try:
                st.json(score_delivery_order(int(oid)))
            except Exception as e:
                st.error(str(e))
    with b:
        cid = st.selectbox("CustomerID", churn["CustomerID"].head(500).astype(int).tolist())
        if st.button("Predict churn"):
            try:
                st.json(score_churn_customer(int(cid)))
            except Exception as e:
                st.error(str(e))
