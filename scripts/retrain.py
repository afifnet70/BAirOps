"""Continuous training with a quality gate.

1. fetch hours newer than the data we have (Open-Meteo) -> data/raw/updates.csv (DVC-versioned by the workflow)
2. train a candidate stacked ensemble (same recipe as Phase 1)
3. compare candidate vs current production on the newest, unseen holdout days
4. promote only if it passes the gate; then refit on all data and register as the new production version
Writes reports/retrain_report.json and sets the GitHub output  promoted=true|false.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bairops.config import PARAMS  # noqa: E402
from bairops.data import load_training_data, read_raw  # noqa: E402
from bairops.drift import save_reference  # noqa: E402
from bairops.features import build_features  # noqa: E402
from bairops.live import fetch_history  # noqa: E402
from bairops.metrics import metrics, very_unhealthy_mae  # noqa: E402
from bairops.model import BAirOpsModel  # noqa: E402
from bairops import registry  # noqa: E402
from bairops.train import make_bundle, refit_all, train_candidate  # noqa: E402

T0 = time.time()
P = PARAMS
NAME = P["project"]["model_name"]
os.makedirs("reports", exist_ok=True)
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def gh_output(key, value):
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"{key}={value}\n")


# ---- 1. new data ---------------------------------------------------------------------------
if os.environ.get("SKIP_FETCH") != "1":
    have = read_raw(P["data"]["raw_zip"])["datetime"].max()
    upd_path = P["data"]["updates_csv"]
    old = pd.read_csv(upd_path, parse_dates=["datetime"]) if os.path.exists(upd_path) else None
    if old is not None and len(old):
        have = max(have, old["datetime"].max())
    start = (have + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if start <= end:
        log(f"fetching new hours {start} .. {end}")
        new = fetch_history(start, end)
        allupd = pd.concat([old, new], ignore_index=True) if old is not None else new
        allupd = allupd.drop_duplicates(["city_id", "datetime"], keep="last").sort_values(["city_id", "datetime"])
        allupd.to_csv(upd_path, index=False)
        log(f"updates.csv now has {len(allupd):,} rows (up to {allupd['datetime'].max()})")
    else:
        log("data already up to date")

# ---- 2. candidate --------------------------------------------------------------------------
df = load_training_data(P)
X, y, meta = build_features(df, P["features"]["horizon_h"], P["data"]["timezone_offset_h"])
log(f"training data: {len(X):,} samples x {X.shape[1]} features, up to {meta['target_time'].max()}")
cand = train_candidate(X, y, meta, P, log=log)
cm = cand["holdout_metrics"]

# ---- 3. production on the same holdout -------------------------------------------------------
path, prod_v, prod_run = registry.download_bundle(NAME)
prod = BAirOpsModel.load(path, version=prod_v)
Xho = X[cand["holdout_mask"]]
pm_pred = prod.predict(Xho)
pm = metrics(cand["holdout_true"], pm_pred)
vu_c, vu_p = very_unhealthy_mae(cand["holdout_true"], cand["holdout_pred"]), very_unhealthy_mae(cand["holdout_true"], pm_pred)
impr = (1 - cm["RMSE"] / pm["RMSE"]) * 100
g = P["promotion"]
vu_ok = not (np.isfinite(vu_c) and np.isfinite(vu_p)) or vu_c <= vu_p * (1 + g["max_very_unhealthy_mae_increase_pct"] / 100)
promote = impr >= g["min_rmse_improvement_pct"] and vu_ok
log(f"holdout RMSE candidate={cm['RMSE']:.3f} vs production v{prod_v}={pm['RMSE']:.3f} ({impr:+.2f}%) | "
    f"AQI>200 MAE {vu_c:.2f} vs {vu_p:.2f} -> {'PROMOTE' if promote else 'KEEP PRODUCTION'}")

report = {"date": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "production_version": prod_v,
          "holdout": [str(cand["hold_start"]), str(cand["end"])], "candidate": cm, "production": pm,
          "candidate_single_models": cand["single_holdout_metrics"], "rmse_improvement_pct": impr,
          "very_unhealthy_mae": {"candidate": vu_c, "production": vu_p}, "gate": g, "promoted": bool(promote)}

# ---- 4. promote ----------------------------------------------------------------------------
if promote:
    final = refit_all(X, y, cand, P, log=log)
    bundle = make_bundle(final, X.columns, cand, P, trained_on=f"all data up to {meta['target_time'].max():%Y-%m-%d}")
    os.makedirs("artifacts", exist_ok=True)
    bpath = "artifacts/bairops_model.joblib"
    joblib.dump(bundle, bpath, compress=3)
    save_reference(df[df["datetime"] >= "2022-08-04"], "reports/reference_stats.json")
    with open("reports/retrain_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    new_v, _ = registry.log_and_register(
        bpath, NAME, metrics={f"holdout_{k}": v for k, v in cm.items()} | {"prod_holdout_RMSE": pm["RMSE"],
                                                                           "rmse_improvement_pct": impr},
        params={"kind": "Ensemble-Stacking", "trained_on": bundle["trained_on"], "previous_version": prod_v,
                **{f"rounds_{k}": v for k, v in cand["rounds"].items()}},
        artifacts=["reports/reference_stats.json", "reports/retrain_report.json"], production=True,
        description=f"auto-retrained; holdout RMSE {cm['RMSE']:.2f} vs v{prod_v} {pm['RMSE']:.2f}",
        run_name="retrain-promoted", tags={"origin": "github-actions-retrain"})
    report["new_version"] = new_v
else:
    import mlflow
    mlflow.set_experiment(registry.EXPERIMENT)
    with mlflow.start_run(run_name="retrain-rejected"):
        mlflow.log_metrics(registry._clean({f"holdout_{k}": v for k, v in cm.items()} |
                                           {"prod_holdout_RMSE": pm["RMSE"], "rmse_improvement_pct": impr}))
        mlflow.set_tags({"origin": "github-actions-retrain", "decision": "kept production", "production_version": prod_v})

with open("reports/retrain_report.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
gh_output("promoted", str(promote).lower())
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(f"## BAirOps retraining\n\n| | Candidate | Production v{prod_v} |\n|---|---|---|\n"
                f"| Holdout RMSE | {cm['RMSE']:.3f} | {pm['RMSE']:.3f} |\n| Holdout MAE | {cm['MAE']:.3f} | {pm['MAE']:.3f} |\n"
                f"| R² | {cm['R2']:.4f} | {pm['R2']:.4f} |\n| Category accuracy | {cm['Category_Acc_%']:.1f}% | {pm['Category_Acc_%']:.1f}% |\n"
                f"| MAE on AQI>200 | {vu_c:.2f} | {vu_p:.2f} |\n\n"
                f"**Decision:** {'promoted to production as v' + str(report.get('new_version')) if promote else 'kept production'} "
                f"(RMSE change {impr:+.2f}%)\n")
log("done")
