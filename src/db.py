from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from .settings import PROCESSED_DIR, ROOT
from .utils import get_logger

log = get_logger("database")
load_dotenv(ROOT / ".env")

TABLE_COLUMNS = {
    "cities": ["CityID", "City", "Population", "Region", "AverageIncome"],
    "customers": ["CustomerID", "Name", "Age", "Gender", "Phone", "Email", "City", "State", "Pincode", "RegistrationDate", "Membership", "TotalOrders", "PreferredCuisine"],
    "restaurants": ["RestaurantID", "RestaurantName", "Cuisine", "City", "Area", "OpeningTime", "ClosingTime", "Rating", "AverageCost", "OwnerName", "RestaurantType", "Latitude", "Longitude"],
    "menu": ["FoodItemID", "RestaurantID", "FoodName", "Category", "Price", "PreparationTime", "Calories", "Availability"],
    "delivery_partners": ["DeliveryPartnerID", "Name", "Age", "Gender", "VehicleType", "JoiningDate", "City", "Rating", "CompletedDeliveries", "AverageDeliveryTime"],
    "promotions": ["PromotionID", "CouponCode", "DiscountPercentage", "CampaignName", "StartDate", "EndDate"],
    "orders": ["OrderID", "CustomerID", "RestaurantID", "DeliveryPartnerID", "OrderDate", "OrderTime", "DeliveryTimeMinutes", "FoodCost", "DeliveryFee", "Discount", "CouponCode", "GST", "FinalAmount", "OrderStatus", "PaymentMethod"],
    "order_items": ["OrderItemID", "OrderID", "FoodItemID", "Quantity", "UnitPrice", "TotalPrice"],
    "payments": ["PaymentID", "OrderID", "PaymentMethod", "PaymentStatus", "TransactionID", "PaymentDate"],
    "customer_feedback": ["FeedbackID", "OrderID", "CustomerRating", "DeliveryRating", "FoodRating", "Review", "Sentiment"],
    "weather": ["WeatherID", "City", "Date", "Temperature", "Rainfall", "Humidity", "WeatherCondition"],
    "traffic": ["TrafficID", "City", "Date", "Time", "TrafficLevel", "AverageSpeed"],
}

LOAD_ORDER = ["cities", "customers", "restaurants", "delivery_partners", "promotions", "menu", "orders", "order_items", "payments", "customer_feedback", "weather", "traffic"]


def load_postgres(database_url: str | None = None, reset_schema: bool = True):
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and edit the connection string.")
    engine = create_engine(url, future=True)

    if reset_schema:
        schema_sql = (ROOT / "sql" / "01_schema_clean.sql").read_text(encoding="utf-8")
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute(schema_sql)
            raw.commit()
            cur.close()
        finally:
            raw.close()
        log.info("Database schema recreated")

    with engine.begin() as conn:
        for table in LOAD_ORDER:
            df = pd.read_pickle(PROCESSED_DIR / f"{table}.pkl")
            cols = TABLE_COLUMNS[table]
            df = df[[c for c in cols if c in df.columns]].copy()
            # SQLAlchemy handles Python date/time objects; convert NaN to None.
            df = df.astype(object).where(pd.notna(df), None)
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
            log.info("Loaded %-20s %7d rows", table, len(df))


if __name__ == "__main__":
    load_postgres()
