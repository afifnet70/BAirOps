"""Data-drift monitoring with the population stability index (PSI), month by month.

Reference distributions are stored per calendar month so that normal seasonality
(winter haze vs monsoon) does not raise false alarms.
"""
import json

import numpy as np
import pandas as pd

DRIFT_VARS = ["aqi", "pm2_5"]


def reference_stats(df, n_bins=10):
    """Decile edges and proportions for each variable and calendar month (local month)."""
    out = {}
    month = (df["datetime"] + pd.Timedelta(hours=6)).dt.month
    for var in DRIFT_VARS:
        out[var] = {}
        for m in range(1, 13):
            v = df.loc[month == m, var].dropna().values
            if len(v) < 100:
                continue
            edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
            if len(edges) < 3:
                continue
            counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
            out[var][str(m)] = {"edges": edges.tolist(), "p": (counts / counts.sum()).tolist()}
    return out


def save_reference(df, path):
    with open(path, "w") as f:
        json.dump(reference_stats(df), f)


def psi(ref, values, eps=1e-4):
    edges, p = np.asarray(ref["edges"]), np.asarray(ref["p"])
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    q = counts / max(counts.sum(), 1)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_report(ref_stats, recent_df):
    """PSI of the recent window against the reference of the same calendar month."""
    month = (recent_df["datetime"] + pd.Timedelta(hours=6)).dt.month
    m = str(int(month.mode().iloc[0]))
    rep = {}
    for var in DRIFT_VARS:
        ref = ref_stats.get(var, {}).get(m)
        rep[f"psi_{var}"] = psi(ref, recent_df[var].dropna().values) if ref else float("nan")
    rep["reference_month"] = int(m)
    return rep
