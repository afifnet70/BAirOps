# BAirOps — End-to-end MLOps for AQI forecasting in Bangladesh

BAirOps forecasts the US Air Quality Index **24 hours ahead** for 30 Bangladeshi locations with a
**stacked ensemble** (Random Forest + XGBoost + LightGBM + CatBoost, non-negative linear meta-learner),
and runs it as a fully automated MLOps system built only on free services.

```
Open-Meteo API ──► ingestion (retry/back-off) ──► shared feature pipeline ──► stacked ensemble
      │                                                                           ▲
      │        GitHub (code) ── DVC (data versions, DagsHub) ── MLflow registry ──┘ (production version)
      │             │
      │             ├─ CI      : tests + Docker build on every push
      │             ├─ CD      : container image -> GitHub Container Registry
      │             ├─ Serve   : hourly batch forecasts -> dashboard + JSON API on GitHub Pages
      │             ├─ Monitor : daily live accuracy + data drift ──► alert issue ──► retrain
      │             ├─ Retrain : weekly; new data -> candidate -> quality gate -> promote / keep
      │             └─ Rollback: one click, any registered version
      ▼
 Dashboard + JSON API (GitHub Pages, refreshed hourly)  ·  FastAPI container for on-demand serving
```

| MLOps capability | Implementation (free) | Where |
|---|---|---|
| Data versioning | DVC, remote storage on DagsHub | `data/raw/*.dvc`, `bootstrap.yml`, `retrain.yml` |
| Experiment tracking | MLflow (hosted on DagsHub) | `src/bairops/registry.py` |
| Model registry + versioning | MLflow model registry, tag `stage=production` | `registry.py` |
| Reproducible training | `params.yaml` + shared code + pinned versions | `src/bairops/train.py` |
| Continuous integration | GitHub Actions: unit tests, Docker build | `.github/workflows/ci.yml` |
| Continuous deployment | GitHub Actions → GitHub Container Registry (image) + GitHub Pages (site) | `deploy.yml`, `pages.yml` |
| Containerisation | multi-stage Dockerfile | `Dockerfile` |
| Serving | hourly batch inference → static dashboard + JSON API; FastAPI container for real-time requests; model always pulled from the registry | `scripts/publish_forecasts.py`, `app/` |
| Continuous training | weekly retrain with promotion gate | `retrain.yml`, `scripts/retrain.py` |
| Monitoring | live RMSE vs observed AQI + PSI drift, auto issue + retrain | `monitor.yml`, `scripts/monitor.py` |
| Rollback | restore any registry version | `rollback.yml`, `scripts/rollback.py` |

---

## Setup (one time, no cost)

1. **DagsHub** (<https://dagshub.com>): sign up with GitHub → **Create → New Repository → Connect a repository →
   GitHub** → select this repository (same name) → copy your token (**avatar → Your Settings → Tokens**).
2. **GitHub secrets** (repo → Settings → Secrets and variables → Actions): `DAGSHUB_USER`, `DAGSHUB_TOKEN`.
3. **Release `v1-assets`** with the dataset zip and the Phase-1 model `bairops_H24.joblib`.
4. **Actions → Bootstrap → Run workflow** (versions the data with DVC, registers the model as v1 = production).
5. **GitHub Pages**: repo → **Settings → Pages → Source: GitHub Actions**.
6. **Actions → Publish forecasts → Run workflow** → the site appears at `https://<user>.github.io/<repo>/`
   and is then refreshed every hour automatically.

Live endpoints (static JSON API): `api/forecasts.json`, `api/city/<city>.json`, `api/model-info.json`, `api/health.json`.
Real-time API (optional, e.g. for latency tests):
```bash
docker run -p 7860:7860 -e MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow \
  -e MLFLOW_TRACKING_USERNAME=<user> -e MLFLOW_TRACKING_PASSWORD=<token> ghcr.io/<user>/bairops
# open http://localhost:7860  (dashboard)  and  http://localhost:7860/docs  (API)
```

---

## Everyday operation
| Want to… | Do |
|---|---|
| change code | push to `main` → CI → new image + site refresh |
| retrain now | Actions → **Retrain** → Run workflow (≈1–2 h; promotes only if better) |
| check live accuracy | Actions → **Monitor** → Run workflow; history in `monitoring/history.csv` |
| roll back | Actions → **Rollback** → Run workflow (empty = previous version); the site switches within minutes |
| see experiments / versions | DagsHub repo → **Experiments** and **Models** tabs |

## Experiments for the paper
1. **Latency/throughput**: start the container (above), then `python scripts/benchmark.py --url http://localhost:7860 --n 200 --concurrency 8`
2. **Continuous training**: run *Retrain*; the job summary shows candidate vs production on the newest 60 unseen days and the decision.
3. **Rollback**: after a v2 exists, run *Rollback* and show `api/model-info.json` switching back to v1.
4. **Live monitoring**: after 2+ weeks, plot `monitoring/history.csv` (live RMSE, PSI, alerts).

## Local development
```bash
pip install -r requirements.txt
PYTHONPATH=src:. pytest -q
BAIROPS_MODEL_PATH=path/to/bairops_H24.joblib PYTHONPATH=src:. uvicorn app.main:app --port 7860
```

## Repository layout
```
params.yaml              all settings (data, features, training, promotion gate, monitoring)
src/bairops/             shared library: data, live ingestion, features, metrics, model, training, registry, drift
app/                     FastAPI service + dashboard
scripts/                 register_model, publish_forecasts, retrain, monitor, rollback, benchmark
site_template/           static dashboard published to GitHub Pages
tests/                   unit + API tests (run in CI)
.github/workflows/       ci, deploy, pages, bootstrap, retrain, monitor, rollback
Dockerfile               multi-stage serving image
```

Model: features use only information available at forecast time (lags, rolling statistics of AQI/PM2.5/PM10,
other pollutants, calendar in Bangladesh time, location). Data: Open-Meteo / CAMS hourly air-quality fields.
