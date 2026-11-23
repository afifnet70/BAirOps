# BAirOps — End-to-end MLOps for near real-time AQI forecasting in Bangladesh

BAirOps forecasts the US Air Quality Index **24 hours ahead** for 30 Bangladeshi locations with a
**stacked ensemble** (Random Forest + XGBoost + LightGBM + CatBoost, non-negative linear meta-learner),
and runs it as a fully automated MLOps system built only on free services.

```
Open-Meteo API ──► ingestion (retry/back-off) ──► shared feature pipeline ──► stacked ensemble
      │                                                                           ▲
      │        GitHub (code) ── DVC (data versions, DagsHub) ── MLflow registry ──┘ (production version)
      │             │
      │             ├─ CI      : tests + Docker build on every push
      │             ├─ CD      : deploy API + dashboard to Hugging Face Spaces
      │             ├─ Monitor : daily live accuracy + data drift ──► alert issue ──► retrain
      │             ├─ Retrain : weekly; new data -> candidate -> quality gate -> promote / keep
      │             └─ Rollback: one click, any registered version
      ▼
 FastAPI service (/forecast, /forecast/all, /predict, /health, /model-info) + city dashboard
```

| MLOps capability | Implementation (free) | Where |
|---|---|---|
| Data versioning | DVC, remote storage on DagsHub | `data/raw/*.dvc`, `bootstrap.yml`, `retrain.yml` |
| Experiment tracking | MLflow (hosted on DagsHub) | `src/bairops/registry.py` |
| Model registry + versioning | MLflow model registry, tag `stage=production` | `registry.py` |
| Reproducible training | `params.yaml` + shared code + pinned versions | `src/bairops/train.py` |
| Continuous integration | GitHub Actions: unit tests, Docker build | `.github/workflows/ci.yml` |
| Continuous deployment | GitHub Actions → Hugging Face Docker Space | `deploy.yml`, `scripts/deploy_space.py` |
| Containerisation | multi-stage Dockerfile | `Dockerfile` |
| Serving | FastAPI + dashboard, model pulled from registry | `app/` |
| Continuous training | weekly retrain with promotion gate | `retrain.yml`, `scripts/retrain.py` |
| Monitoring | live RMSE vs observed AQI + PSI drift, auto issue + retrain | `monitor.yml`, `scripts/monitor.py` |
| Rollback | restore any registry version | `rollback.yml`, `scripts/rollback.py` |

---

## Setup (one time, about 30 minutes, no cost)

### 1. Create two free accounts and tokens
1. **DagsHub** (<https://dagshub.com>): *Sign up with GitHub* → **Create → New Repository → Connect a repository →
   GitHub** → select **this repository** → *Connect* (keep the same repository name). Then **avatar → Your Settings → Tokens** and copy your token.
2. **Hugging Face** (<https://huggingface.co>): sign up → **Settings → Access Tokens → Create new token**,
   type **Write** → copy it.

### 2. Put this code in your GitHub repository
```bash
git clone https://github.com/<your-github-user>/<your-repo>.git
cd <your-repo>
# copy ALL files of this folder here (including the hidden .github, .dvc, .gitignore, .dockerignore, .dvcignore)
git add -A
git commit -m "BAirOps MLOps framework"
git push
```
The **CI** workflow starts automatically (tab **Actions**). It should finish green.

### 3. Add 4 secrets
GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|---|---|
| `DAGSHUB_USER` | your DagsHub username |
| `DAGSHUB_TOKEN` | the DagsHub token |
| `HF_TOKEN` | the Hugging Face write token |
| `HF_SPACE` | `<your-hf-username>/bairops` |

### 4. Upload the dataset and the Phase-1 model as a GitHub Release
GitHub repo → **Releases → Draft a new release** → tag **`v1-assets`** → attach:
* the dataset zip (`Bangladesh_Air_Quality_Index_AQI_Dataset_2000_2...zip`)
* `bairops_H24.joblib` (from the Phase-1 Colab output, folder `models/`)

→ **Publish release**.

### 5. Run the Bootstrap workflow (once)
**Actions → Bootstrap → Run workflow**. It (a) versions the dataset with DVC on DagsHub, (b) registers the
Phase-1 stacked ensemble as **`bairops-stacking` v1 = production** in MLflow, (c) deploys the app.

### 6. Open the live system
`https://huggingface.co/spaces/<your-hf-username>/bairops` (first build ≈ 10 min).
Dashboard at `/`, interactive API docs at `/docs`, status at `/health`, model details at `/model-info`.

That's it — monitoring runs **daily**, retraining **weekly**, and both can be started by hand in **Actions**.

---

## Everyday operation
| Want to… | Do |
|---|---|
| change code | push to `main` → CI → automatic deploy |
| retrain now | Actions → **Retrain** → Run workflow (≈1–2 h; promotes only if better) |
| check live accuracy | Actions → **Monitor** → Run workflow; history in `monitoring/history.csv` |
| roll back | Actions → **Rollback** → Run workflow (empty = previous version) |
| see experiments / versions | DagsHub repo → **Experiments** and **Models** tabs |

The free Hugging Face Space sleeps after 48 h without visitors and wakes on the next visit (~1 min).

## Experiments for the paper
1. **Latency/throughput**: `python scripts/benchmark.py --url https://<hf-user>-bairops.hf.space --n 200 --concurrency 8`
2. **Continuous training**: run *Retrain*; the job summary shows candidate vs production on the newest 60 unseen days and the decision.
3. **Rollback**: after a v2 exists, run *Rollback* and show `/model-info` switching back to v1.
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
scripts/                 register_model, retrain, monitor, rollback, deploy_space, benchmark
tests/                   unit + API tests (run in CI)
.github/workflows/       ci, deploy, bootstrap, retrain, monitor, rollback
Dockerfile               multi-stage serving image
```

Model: features use only information available at forecast time (lags, rolling statistics of AQI/PM2.5/PM10,
other pollutants, calendar in Bangladesh time, location). Data: Open-Meteo / CAMS hourly air-quality fields.
