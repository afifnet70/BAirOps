"""MLflow tracking + model registry helpers (hosted free on DagsHub).

Environment variables (set as GitHub / Space secrets):
  MLFLOW_TRACKING_URI       https://dagshub.com/<user>/<repo>.mlflow
  MLFLOW_TRACKING_USERNAME  <dagshub user>
  MLFLOW_TRACKING_PASSWORD  <dagshub token>

The deployed version is marked with the model-version tag  stage=production  (older ones get
stage=archived), and the registry alias "production" is set as well when the server supports it.
"""
import glob
import os
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT = "BAirOps"


def _client():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI is not set (see README: DagsHub secrets).")
    return MlflowClient()


def _clean(d):
    return {k.replace("%", "pct"): float(v) for k, v in (d or {}).items() if v == v}


def log_and_register(bundle_path, model_name, metrics=None, params=None, artifacts=(), production=False,
                     description="", run_name="register", tags=None):
    """Log a run (params, metrics, artifacts) and register the bundle as a new model version."""
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        if metrics:
            mlflow.log_metrics(_clean(metrics))
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_artifact(bundle_path, artifact_path="model")
        for a in artifacts:
            if a and os.path.exists(a):
                mlflow.log_artifact(a, artifact_path="reports")
        run_id, art_uri = run.info.run_id, run.info.artifact_uri
    c = _client()
    try:
        c.create_registered_model(model_name, description="BAirOps stacked-ensemble AQI forecaster (24 h ahead)")
    except Exception:
        pass                                            # already exists
    mv = c.create_model_version(name=model_name, source=f"{art_uri}/model", run_id=run_id, description=description)
    for k, v in _clean(metrics).items():
        c.set_model_version_tag(model_name, mv.version, k, f"{v:.4f}")
    c.set_model_version_tag(model_name, mv.version, "stage", "candidate")
    if production:
        promote(model_name, mv.version)
    print(f"[registry] registered {model_name} v{mv.version} (run {run_id})")
    return mv.version, run_id


def versions(model_name):
    vs = _client().search_model_versions(f"name='{model_name}'")
    return sorted(vs, key=lambda v: int(v.version))


def production_version(model_name):
    prod = [v for v in versions(model_name) if (v.tags or {}).get("stage") == "production"]
    if not prod:
        raise RuntimeError(f"No production version of '{model_name}' in the registry yet.")
    return prod[-1]


def promote(model_name, version):
    """Mark `version` as production; all other production versions become archived."""
    c = _client()
    version = str(version)
    for v in versions(model_name):
        if (v.tags or {}).get("stage") == "production" and str(v.version) != version:
            c.set_model_version_tag(model_name, v.version, "stage", "archived")
    c.set_model_version_tag(model_name, version, "stage", "production")
    try:
        c.set_registered_model_alias(model_name, "production", version)
    except Exception:
        pass                                            # aliases not supported by every server
    print(f"[registry] {model_name} v{version} -> production")


def download_bundle(model_name, version=None, dst=None):
    """Download the joblib bundle of a version (default: production). Returns (path, version)."""
    v = production_version(model_name) if version is None else _client().get_model_version(model_name, str(version))
    dst = dst or tempfile.mkdtemp(prefix="bairops_")
    local = mlflow.artifacts.download_artifacts(run_id=v.run_id, artifact_path="model", dst_path=dst)
    files = glob.glob(os.path.join(local, "**", "*.joblib"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No .joblib in the artifacts of {model_name} v{v.version}")
    return files[0], str(v.version), v.run_id


def download_report(run_id, name, dst=None):
    """Download reports/<name> of a run; returns the local path or None."""
    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"reports/{name}",
                                                   dst_path=dst or tempfile.mkdtemp(prefix="bairops_"))
    except Exception:
        return None
