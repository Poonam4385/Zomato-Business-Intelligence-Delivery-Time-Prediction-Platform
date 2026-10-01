from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .settings import CITY_ALIASES, CFG


def get_logger(name: str = "zomato") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def normalize_city(value):
    if pd.isna(value):
        return np.nan
    s = re.sub(r"\s+", " ", str(value)).strip()
    if not s:
        return np.nan
    # title-case only when input is all lower/upper; preserve Bengaluru etc.
    if s.islower() or s.isupper():
        s = s.title()
    return CITY_ALIASES.get(s, s)


def _build_date(year: int, month: int, day: int):
    try:
        return pd.Timestamp(year=year, month=month, day=day).normalize()
    except Exception:
        return pd.NaT


def parse_mixed_date_value(value, ambiguous_policy: str | None = None):
    """Parse intentionally mixed dates while explicitly flagging ambiguity.

    Returns (timestamp, ambiguous_flag, parse_failed_flag, format_label).
    Ambiguous numeric dates such as 03/04/2024 cannot be recovered uniquely;
    the configured policy is used while a flag is retained for auditability.
    """
    if pd.isna(value):
        return pd.NaT, False, False, "missing"
    s = str(value).strip()
    if not s:
        return pd.NaT, False, False, "missing"
    ambiguous_policy = ambiguous_policy or CFG["data"].get("ambiguous_date_policy", "dayfirst")

    # Unambiguous year-first formats: YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD
    m = re.fullmatch(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        ts = _build_date(y, mo, d)
        return ts, False, pd.isna(ts), "ymd"

    # Day/month/year or month/day/year numeric forms.
    m = re.fullmatch(r"(\d{1,2})[-./](\d{1,2})[-./](\d{4})", s)
    if m:
        a, b, y = map(int, m.groups())
        if a > 12 and b <= 12:  # definitely day first
            ts = _build_date(y, b, a)
            return ts, False, pd.isna(ts), "dmy"
        if b > 12 and a <= 12:  # definitely month first
            ts = _build_date(y, a, b)
            return ts, False, pd.isna(ts), "mdy"
        if a <= 12 and b <= 12:
            if ambiguous_policy == "monthfirst":
                ts = _build_date(y, a, b)
                label = "ambiguous_mdy_assumed"
            else:
                ts = _build_date(y, b, a)
                label = "ambiguous_dmy_assumed"
            return ts, True, pd.isna(ts), label

    # Last-resort parser for textual dates, kept conservative.
    try:
        ts = pd.to_datetime(s, errors="raise")
        return pd.Timestamp(ts).normalize(), False, False, "fallback"
    except Exception:
        return pd.NaT, False, True, "failed"


def parse_mixed_date_series(series: pd.Series, prefix: str):
    parsed = series.map(parse_mixed_date_value)
    out = pd.DataFrame(parsed.tolist(), index=series.index,
                       columns=[prefix, f"{prefix}AmbiguousFlag", f"{prefix}ParseFailedFlag", f"{prefix}ParseFormat"])
    return out


def parse_time_series(series: pd.Series) -> pd.Series:
    x = pd.to_datetime(series.astype(str).str.strip(), format="mixed", errors="coerce")
    return x.dt.time


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl":
        df.to_pickle(path)
    elif path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError:
            df.to_pickle(path.with_suffix(".pkl"))
    else:
        df.to_csv(path, index=False)


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))
