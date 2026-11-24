"""Batch serving: produce the latest 24-h forecasts for all cities with the PRODUCTION model from the
MLflow registry and write a static site (dashboard + JSON API) for GitHub Pages.

  python scripts/publish_forecasts.py --out site
Outputs:
  site/index.html                 dashboard
  site/api/forecasts.json         all cities (incl. last 7 days of observed AQI)
  site/api/city/<name>.json       one file per city
  site/api/model-info.json        production model version + test metrics
  site/api/health.json            run status / timestamps / latency
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]

from app.service import Service  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="site")
a = ap.parse_args()

t0 = time.time()
svc = Service()
svc.start(background=False)                       # loads production model from the registry, fetches live data, predicts
out_api = os.path.join(a.out, "api")
os.makedirs(os.path.join(out_api, "city"), exist_ok=True)

full = []
for name in svc.cities():
    f = svc.forecast(name)
    if f is None:
        continue
    full.append(f)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    with open(os.path.join(out_api, "city", f"{slug}.json"), "w") as fh:
        json.dump(f, fh)
full.sort(key=lambda f: -f["aqi_forecast"])

info = svc.model.info()
health = dict(svc.health(), generated_utc=svc.last_refresh, run_seconds=round(time.time() - t0, 1),
              serving="batch (hourly GitHub Actions job)", git_sha=os.environ.get("GITHUB_SHA", "local")[:7])
with open(os.path.join(out_api, "forecasts.json"), "w") as fh:
    json.dump({"generated_utc": svc.last_refresh, "model_version": svc.model.version,
               "explain_global": svc.explain_global, "forecasts": full}, fh)
with open(os.path.join(out_api, "model-info.json"), "w") as fh:
    json.dump(info, fh, indent=2)
with open(os.path.join(out_api, "health.json"), "w") as fh:
    json.dump(health, fh, indent=2)
shutil.copy(os.path.join(ROOT, "site_template", "index.html"), os.path.join(a.out, "index.html"))
open(os.path.join(a.out, ".nojekyll"), "w").close()
# explainable-AI log (used for the paper): global driver ranking and four demonstration cities
print("XAI_GLOBAL " + json.dumps(svc.explain_global))
for f in full:
    if f["city"] in ("Dhaka", "Chittagong", "Rajshahi", "Sylhet") and f.get("explanation"):
        print("XAI_CITY " + json.dumps({"city": f["city"], "now": f["aqi_now"], **f["explanation"]}))
print(f"published {len(full)} city forecasts with model v{svc.model.version} in {time.time() - t0:.1f}s")
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
        fh.write(f"## BAirOps forecasts published\n\nModel v{svc.model.version} · {len(full)} cities · "
                 f"generated {svc.last_refresh} UTC\n\n| City | Now | Forecast (+24 h) | Category |\n|---|---|---|---|\n" +
                 "\n".join(f"| {f['city']} | {f['aqi_now']:.0f} | {f['aqi_forecast']:.0f} | {f['category_forecast']} |" for f in full[:10]) + "\n")
