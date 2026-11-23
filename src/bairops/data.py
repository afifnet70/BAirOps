"""Data loading and cleaning (identical rules to the Phase-1 notebook)."""
import glob
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd

from .features import POLLUTANTS

PHYS_MAX = {"pm10": 2000, "pm2_5": 1500, "carbon_monoxide": 50000, "nitrogen_dioxide": 2000,
            "sulphur_dioxide": 2000, "ozone": 1000, "aqi": 500}
COLUMNS = ["city_id", "city_name", "lat", "lon", "datetime"] + POLLUTANTS + ["aqi"]


def read_raw(zip_or_csv):
    """Read 'AQI Bangladesh.csv' from the dataset zip (or a csv path)."""
    if str(zip_or_csv).lower().endswith(".csv"):
        df = pd.read_csv(zip_or_csv)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_or_csv) as zf:
                zf.extractall(tmp)
            csvs = [p for p in glob.glob(os.path.join(tmp, "**", "*.csv"), recursive=True)
                    if "cities" not in os.path.basename(p).lower()]
            if not csvs:
                raise FileNotFoundError("No AQI csv inside the zip")
            df = pd.read_csv(csvs[0])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def clean(raw, valid_start="2022-08-04", include_dhaka_2000_2022=True, max_gap_h=6):
    """Valid period, physical limits, complete hourly grid per city, short-gap interpolation."""
    keep = raw["datetime"] >= pd.Timestamp(valid_start)
    if include_dhaka_2000_2022:
        keep |= raw["city_name"].astype(str).str.lower().eq("dhaka")
    df = raw[keep][[c for c in COLUMNS if c in raw.columns]].copy()
    df = df.drop_duplicates(["city_id", "datetime"], keep="last").sort_values(["city_id", "datetime"])
    for c, mx in PHYS_MAX.items():
        df.loc[(df[c] < 0) | (df[c] > mx), c] = np.nan
    parts = []
    num = POLLUTANTS + ["aqi"]
    for cid, g in df.groupby("city_id", sort=True):
        g = g.set_index("datetime")
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        g[num] = g[num].interpolate(method="linear", limit=max_gap_h, limit_area="inside")
        g["city_id"] = cid
        for c in ("city_name", "lat", "lon"):
            g[c] = g[c].ffill().bfill()
        parts.append(g.rename_axis("datetime").reset_index())
    return pd.concat(parts, ignore_index=True)[COLUMNS]


def load_training_data(params):
    """Original DVC dataset + (optional) newer hours appended by the retrain job."""
    p = params["data"]
    raw = read_raw(p["raw_zip"])
    if p.get("updates_csv") and os.path.exists(p["updates_csv"]):
        upd = pd.read_csv(p["updates_csv"], parse_dates=["datetime"])
        raw = pd.concat([raw, upd], ignore_index=True)
    return clean(raw, p["valid_start"], p["include_dhaka_2000_2022"], p["max_interp_gap_h"])
