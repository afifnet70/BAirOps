"""Live / historical data ingestion from the Open-Meteo Air Quality API (free, no key).

Robust ingestion as in GreenAirOps: retries with exponential back-off + jitter, rate-limit
aware waiting (HTTP 429), and a short-lived cache in the API layer.
"""
import random
import time

import numpy as np
import pandas as pd
import requests

from .cities import CITIES
from .features import POLLUTANTS

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
API_VARS = POLLUTANTS + ["us_aqi"]


def _get(params, retries=6, timeout=60):
    last = None
    for k in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 429:                      # rate limited -> wait a full minute
                time.sleep(65)
                continue
            if 400 <= r.status_code < 500:
                break
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(min(60, 2 ** k + random.random()))
    raise RuntimeError(f"Open-Meteo request failed: {last}")


def _to_frame(js, city):
    h = js["hourly"]
    df = pd.DataFrame({"datetime": pd.to_datetime(h["time"])})
    for v in API_VARS:
        df[v] = pd.to_numeric(pd.Series(h.get(v, [np.nan] * len(df))), errors="coerce").values
    df = df.rename(columns={"us_aqi": "aqi"})
    df["city_id"], df["city_name"], df["lat"], df["lon"] = city["city_id"], city["name"], city["lat"], city["lon"]
    return df


def _only_observed(df):
    """Drop hours after the current UTC hour (the API also returns forecast hours)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    return df[df["datetime"] <= now]


def fetch_recent(cities=None, past_days=9):
    """Last `past_days` days of hourly data for all cities, in ONE multi-location request."""
    cities = cities or CITIES
    params = {"latitude": ",".join(str(c["lat"]) for c in cities),
              "longitude": ",".join(str(c["lon"]) for c in cities),
              "hourly": ",".join(API_VARS), "past_days": past_days, "forecast_days": 1, "timezone": "GMT"}
    try:
        js = _get(params)
        js = js if isinstance(js, list) else [js]
        frames = [_to_frame(j, c) for j, c in zip(js, cities)]
    except Exception:
        frames = []                                         # fall back to one request per city
        for c in cities:
            p = dict(params, latitude=c["lat"], longitude=c["lon"])
            frames.append(_to_frame(_get(p), c))
            time.sleep(0.3)
    return _only_observed(pd.concat(frames, ignore_index=True))


def fetch_history(start, end, cities=None, pause_s=6.0):
    """Hourly history between two dates (YYYY-MM-DD), one city per request, paced for the free tier."""
    cities = cities or CITIES
    frames = []
    for i, c in enumerate(cities, 1):
        p = {"latitude": c["lat"], "longitude": c["lon"], "hourly": ",".join(API_VARS),
             "start_date": start, "end_date": end, "timezone": "GMT"}
        frames.append(_to_frame(_get(p), c))
        print(f"  history {i:2d}/{len(cities)} {c['name']}")
        time.sleep(pause_s)
    return _only_observed(pd.concat(frames, ignore_index=True))
