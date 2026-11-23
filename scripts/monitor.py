"""Daily monitoring of the production model on live data.

* live performance: re-creates the 24-h forecasts issued during the last `window_days` and compares
  them with the AQI that was actually observed (no prediction log needed);
* data drift: month-matched population stability index (PSI) of AQI and PM2.5.
Appends a row to monitoring/history.csv, writes monitoring/latest.json and sets the GitHub output alert=true|false.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bairops import registry  # noqa: E402
from bairops.config import PARAMS  # noqa: E402
from bairops.data import clean  # noqa: E402
from bairops.drift import drift_report  # noqa: E402
from bairops.features import build_features  # noqa: E402
from bairops.live import fetch_recent  # noqa: E402
from bairops.metrics import metrics  # noqa: E402
from bairops.model import BAirOpsModel  # noqa: E402

P, M = PARAMS, PARAMS["monitoring"]
NAME = P["project"]["model_name"]
os.makedirs("monitoring", exist_ok=True)

path, version, run_id = registry.download_bundle(NAME)
model = BAirOpsModel.load(path, version=version)
ref_test_rmse = float((model.bundle.get("test_metrics") or {}).get("RMSE", np.nan))

recent = fetch_recent(past_days=M["window_days"] + 9)
data = clean(recent, valid_start="2000-01-01", include_dhaka_2000_2022=True, max_gap_h=P["data"]["max_interp_gap_h"])
X, y, meta = build_features(data, model.horizon_h, P["data"]["timezone_offset_h"])
now = data["datetime"].max()
win = (meta["target_time"] > now - pd.Timedelta(days=M["window_days"])).values
pred = model.predict(X[win])
live = metrics(y[win].values, pred)

ref_path = registry.download_report(run_id, "reference_stats.json")
if ref_path:
    with open(ref_path) as f:
        drift = drift_report(json.load(f), data[data["datetime"] > now - pd.Timedelta(days=M["window_days"])])
else:
    drift = {"psi_aqi": float("nan"), "psi_pm2_5": float("nan"), "reference_month": None}

alert_perf = np.isfinite(ref_test_rmse) and live["RMSE"] > M["rmse_alert_factor"] * ref_test_rmse
alert_drift = any(np.isfinite(drift[k]) and drift[k] > M["psi_alert"] for k in ("psi_aqi", "psi_pm2_5"))
row = {"date": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"), "model_version": version,
       "window_end": str(now), "n_forecasts": int(win.sum()),
       "live_RMSE": round(live["RMSE"], 3), "live_MAE": round(live["MAE"], 3), "live_R2": round(live["R2"], 4),
       "live_Category_Acc_%": round(live["Category_Acc_%"], 2), "reference_test_RMSE": round(ref_test_rmse, 3),
       "psi_aqi": round(drift["psi_aqi"], 4), "psi_pm2_5": round(drift["psi_pm2_5"], 4),
       "alert_performance": bool(alert_perf), "alert_drift": bool(alert_drift)}
# per-city live evaluation + the full backtest (used for the paper's live demonstration)
lo, hi = model.interval(pred)
bt = meta[win][["city_name", "issue_time", "target_time"]].reset_index(drop=True)
bt["observed"], bt["predicted"], bt["pi_low"], bt["pi_high"] = y[win].values, pred, lo, hi
bt["persistence"] = meta[win]["aqi_now"].values
bt.round(2).to_csv("monitoring/backtest_latest.csv", index=False)
rows_c = []
for c, g in bt.groupby("city_name"):
    m_ = metrics(g["observed"], g["predicted"])
    rows_c.append({"city": c, "n": len(g), "RMSE": round(m_["RMSE"], 3), "MAE": round(m_["MAE"], 3),
                   "Bias": round(m_["Bias"], 3), "Category_Acc_%": round(m_["Category_Acc_%"], 1),
                   "persistence_RMSE": round(metrics(g["observed"], g["persistence"])["RMSE"], 3),
                   "PI90_coverage_%": round(100 * float(((g["observed"] >= g["pi_low"]) & (g["observed"] <= g["pi_high"])).mean()), 1)})
pd.DataFrame(rows_c).to_csv("monitoring/latest_by_city.csv", index=False)
live_persist = metrics(bt["observed"], bt["persistence"])
row["live_persistence_RMSE"] = round(live_persist["RMSE"], 3)

hist_path = "monitoring/history.csv"
hist = pd.read_csv(hist_path) if os.path.exists(hist_path) and os.path.getsize(hist_path) > 0 else pd.DataFrame()
pd.concat([hist, pd.DataFrame([row])], ignore_index=True).to_csv(hist_path, index=False)
with open("monitoring/latest.json", "w") as f:
    json.dump(row, f, indent=2)
print(json.dumps(row, indent=2))

alert = alert_perf or alert_drift
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"alert={str(alert).lower()}\n")
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(f"## BAirOps monitoring (model v{version})\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in row.items()) + "\n")
