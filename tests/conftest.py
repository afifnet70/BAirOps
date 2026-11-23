import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]

from bairops.cities import CITIES  # noqa: E402
from bairops.features import build_features  # noqa: E402


def synthetic(n_hours=420, cities=None, seed=0):
    """Hourly data that looks like the real feed (UTC, pollutants, US AQI)."""
    rng = np.random.default_rng(seed)
    cities = cities or CITIES[:3]
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    t = pd.date_range(end=end, periods=n_hours, freq="h")
    out = []
    for k, c in enumerate(cities):
        base = 80 + 30 * k + 25 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + rng.normal(0, 5, n_hours).cumsum() * 0.1
        pm25 = np.clip(base * 0.5, 1, None)
        out.append(pd.DataFrame({
            "city_id": c["city_id"], "city_name": c["name"], "lat": c["lat"], "lon": c["lon"], "datetime": t,
            "pm10": pm25 * 1.4, "pm2_5": pm25, "carbon_monoxide": 400 + pm25 * 3, "nitrogen_dioxide": 20 + pm25 * 0.2,
            "sulphur_dioxide": 10 + pm25 * 0.1, "ozone": 40 + 10 * np.cos(np.arange(n_hours) * 2 * np.pi / 24),
            "aqi": np.clip(base, 5, 400)}))
    return pd.concat(out, ignore_index=True)


@pytest.fixture(scope="session")
def data():
    return synthetic()


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory, data):
    """A tiny stand-in for the real stacked-ensemble bundle (same structure, sklearn models)."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    X, y, _ = build_features(data, 24, 6)
    models = {"RandomForest": DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y),
              "XGBoost": Ridge(alpha=1.0).fit(X, y), "LightGBM": Ridge(alpha=10.0).fit(X, y),
              "CatBoost": DecisionTreeRegressor(max_depth=6, random_state=0).fit(X, y)}
    P = np.column_stack([m.predict(X) for m in models.values()])
    b = {"models": models, "model_order": list(models), "features": list(X.columns), "kind": "Ensemble-Stacking",
         "selected": "Ensemble-Stacking", "stacker": LinearRegression(positive=True).fit(P, y), "weights": None,
         "pi_halfwidth": 25.0, "horizon_h": 24, "trained_on": "synthetic", "test_metrics": {"RMSE": 16.28, "MAE": 11.96}}
    p = tmp_path_factory.mktemp("model") / "bundle.joblib"
    joblib.dump(b, p)
    return str(p)
