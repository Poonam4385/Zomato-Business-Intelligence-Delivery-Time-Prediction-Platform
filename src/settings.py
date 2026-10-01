from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


CFG = load_config()
RAW_DIR = ROOT / CFG["data"]["raw_dir"]
PROCESSED_DIR = ROOT / CFG["data"]["processed_dir"]
FEATURE_DIR = ROOT / CFG["data"]["feature_dir"]
REPORT_DIR = ROOT / CFG["data"]["report_dir"]
MODEL_DIR = ROOT / CFG["data"]["model_dir"]

for p in [RAW_DIR, PROCESSED_DIR, FEATURE_DIR, REPORT_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "cities": "cities (1).csv",
    "customers": "customers (2).csv",
    "restaurants": "restaurants.csv",
    "menu": "menu.csv",
    "delivery_partners": "delivery_partners.csv",
    "promotions": "promotions.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "payments": "payments (1).csv",
    "customer_feedback": "customer_feedback (1).csv",
    "weather": "weather.csv",
    "traffic": "traffic.csv",
}

COMPLETED_STATUSES = set(CFG["completed_statuses"])
CITY_ALIASES = CFG.get("city_aliases", {})
RANDOM_STATE = int(CFG["project"]["random_state"])
