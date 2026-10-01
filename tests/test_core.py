from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import parse_mixed_date_value


def test_mixed_date_unambiguous_dmy():
    ts, ambiguous, failed, label = parse_mixed_date_value("18/03/2024")
    assert ts == pd.Timestamp("2024-03-18")
    assert ambiguous is False
    assert failed is False
    assert label == "dmy"


def test_mixed_date_unambiguous_mdy():
    ts, ambiguous, failed, label = parse_mixed_date_value("03/18/2024")
    assert ts == pd.Timestamp("2024-03-18")
    assert ambiguous is False
    assert failed is False
    assert label == "mdy"


def test_mixed_date_ambiguity_is_flagged():
    ts, ambiguous, failed, label = parse_mixed_date_value("03/04/2024", ambiguous_policy="dayfirst")
    assert ts == pd.Timestamp("2024-04-03")
    assert ambiguous is True
    assert failed is False
    assert "ambiguous" in label


def test_financial_equation_in_processed_data():
    orders = pd.read_pickle("data/processed/orders.pkl")
    cols = ["FoodCost", "DeliveryFee", "Discount", "GST", "FinalAmount"]
    x = orders.dropna(subset=cols)
    expected = x["FoodCost"] + x["DeliveryFee"] - x["Discount"] + x["GST"]
    assert np.allclose(expected, x["FinalAmount"], atol=0.01)


def test_primary_keys_unique():
    specs = {
        "orders": "OrderID", "customers": "CustomerID", "restaurants": "RestaurantID",
        "delivery_partners": "DeliveryPartnerID", "menu": "FoodItemID", "order_items": "OrderItemID",
    }
    for table, key in specs.items():
        df = pd.read_pickle(f"data/processed/{table}.pkl")
        assert not df[key].duplicated().any(), f"duplicate key in {table}"
