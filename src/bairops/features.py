"""Feature engineering shared by training, the live API and monitoring.

Identical to the Phase-1 notebook (pollutant-only model): only information available at
issue time t is used to predict AQI at t + horizon.
"""
import numpy as np
import pandas as pd

POLLUTANTS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
SEASON_NAMES = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
HISTORY_HOURS = 168            # longest look-back used by any feature


def bd_season(month):
    """Winter (Dec-Feb)=0, Pre-monsoon (Mar-May)=1, Monsoon (Jun-Sep)=2, Post-monsoon (Oct-Nov)=3."""
    month = np.asarray(month)
    return np.select([np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8, 9])],
                     [0, 1, 2], default=3)


def build_features(d, horizon=24, tz_offset_h=6, for_inference=False):
    """d: hourly rows with city_id, city_name, lat, lon, datetime (UTC), pollutants and aqi.

    Returns (X, y, meta). With for_inference=True rows without a known target are kept
    (y is NaN there), which is what the live service needs for the newest hour.
    """
    d = d.sort_values(["city_id", "datetime"]).reset_index(drop=True)
    g = d.groupby("city_id", sort=False)
    F = pd.DataFrame(index=d.index)
    F["lat"], F["lon"] = d["lat"], d["lon"]
    for c in POLLUTANTS + ["aqi"]:
        F[c] = d[c]
    for c in ["aqi", "pm2_5", "pm10"]:
        for L in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            F[f"{c}_lag{L}"] = g[c].shift(L)
        for w in [3, 6, 12, 24, 72, 168]:
            F[f"{c}_rmean{w}"] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        for w in [6, 24, 72]:
            F[f"{c}_rstd{w}"] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
    for c in ["nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide"]:
        F[f"{c}_lag24"] = g[c].shift(24)
        F[f"{c}_rmean24"] = g[c].transform(lambda s: s.rolling(24, min_periods=24).mean())
    F["aqi_diff1"] = d["aqi"] - F["aqi_lag1"]
    F["aqi_diff24"] = d["aqi"] - F["aqi_lag24"]
    F["pm_ratio"] = d["pm2_5"] / (d["pm10"] + 1e-3)

    target_time = d["datetime"] + pd.Timedelta(hours=horizon)
    local = target_time + pd.Timedelta(hours=tz_offset_h)
    hr, doy, dow, mon = local.dt.hour, local.dt.dayofyear, local.dt.dayofweek, local.dt.month
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    F["doy_sin"], F["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    F["hour"], F["dayofweek"], F["month"] = hr, dow, mon
    F["is_weekend_bd"] = dow.isin([4, 5]).astype(int)          # Friday-Saturday
    F["season"] = bd_season(mon.values)

    y = g["aqi"].shift(-horizon)
    meta = pd.DataFrame({"city_id": d["city_id"], "city_name": d["city_name"], "issue_time": d["datetime"],
                         "target_time": target_time, "aqi_now": d["aqi"], "season": F["season"]})
    ok = F.notna().all(axis=1)
    if not for_inference:
        ok &= y.notna()
    return (F[ok].astype("float32").reset_index(drop=True), y[ok].astype("float32").reset_index(drop=True),
            meta[ok].reset_index(drop=True))
