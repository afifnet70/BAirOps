"""Register the Phase-1 stacked ensemble (trained in Colab) as version 1 = production.

Usage (run by the Bootstrap workflow):
  python scripts/register_model.py --bundle assets/bairops_H24.joblib --data data/raw/aqi_bangladesh.zip
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib  # noqa: E402

from bairops.config import PARAMS  # noqa: E402
from bairops.data import clean, read_raw  # noqa: E402
from bairops.drift import save_reference  # noqa: E402
from bairops.registry import log_and_register  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
ap.add_argument("--data", default=PARAMS["data"]["raw_zip"])
ap.add_argument("--description", default="v1: Phase-1 stacked ensemble trained in Colab (test year 2024-11-24..2025-11-23)")
a = ap.parse_args()

b = joblib.load(a.bundle)
assert b.get("kind") == "Ensemble-Stacking" or b.get("selected") == "Ensemble-Stacking", \
    f"expected the stacked ensemble, got {b.get('kind')}/{b.get('selected')}"
print("bundle:", b.get("selected"), "| features:", len(b["features"]), "| trained on:", b.get("trained_on"))

os.makedirs("reports", exist_ok=True)
p = PARAMS["data"]
df = clean(read_raw(a.data), p["valid_start"], p["include_dhaka_2000_2022"], p["max_interp_gap_h"])
save_reference(df[df["datetime"] >= "2022-08-04"], "reports/reference_stats.json")
card = {"name": "BAirOps stacked ensemble", "horizon_h": b.get("horizon_h"), "kind": b.get("kind"),
        "base_models": b.get("model_order"), "n_features": len(b["features"]), "trained_on": b.get("trained_on"),
        "test_period": "2024-11-24 .. 2025-11-23", "test_metrics": b.get("test_metrics"),
        "pi90_halfwidth": b.get("pi_halfwidth"), "library_versions": b.get("versions")}
with open("reports/model_card.json", "w") as f:
    json.dump(card, f, indent=2, default=str)

version, run_id = log_and_register(
    a.bundle, PARAMS["project"]["model_name"],
    metrics={f"test_{k}": v for k, v in (b.get("test_metrics") or {}).items()} | {"pi90_halfwidth": b.get("pi_halfwidth", float("nan"))},
    params={"kind": b.get("kind"), "horizon_h": b.get("horizon_h"), "n_features": len(b["features"]),
            "trained_on": b.get("trained_on"), "origin": "phase1-colab"},
    artifacts=["reports/reference_stats.json", "reports/model_card.json"],
    production=True, description=a.description, run_name="v1-phase1-colab",
    tags={"origin": "phase1-colab"})
print(f"OK: registered as version {version} (production)")
