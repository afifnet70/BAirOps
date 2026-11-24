import numpy as np

from bairops.features import build_features
from bairops.metrics import aqi_cat, aqi_label, metrics, very_unhealthy_mae
from bairops.model import BAirOpsModel


def test_aqi_categories():
    assert list(aqi_cat([0, 50, 51, 100, 150, 200, 300, 301])) == [0, 0, 1, 1, 2, 3, 4, 5]
    assert aqi_label(160) == "Unhealthy"


def test_metrics_perfect():
    m = metrics([10, 60, 120, 250], [10, 60, 120, 250])
    assert m["RMSE"] == 0 and m["Category_Acc_%"] == 100
    assert np.isnan(very_unhealthy_mae([10, 20], [11, 21]))


def test_bundle_prediction(bundle_path, data):
    model = BAirOpsModel.load(bundle_path, version="7")
    X, y, _ = build_features(data, 24, 6)
    p = model.predict(X)
    assert p.shape == (len(X),) and (p >= 0).all() and (p <= 500).all()
    lo, hi = model.interval(p)
    assert (np.isclose(hi - lo, 50) | (lo == 0) | (hi == 500)).all()
    info = model.info()
    assert info["version"] == "7" and info["kind"] == "Ensemble-Stacking" and len(info["base_models"]) == 4


def test_explanations_add_up(bundle_path, data):
    """base + sum of contributions must equal the forecast (stand-in sklearn learners)."""
    from bairops.explain import DRIVERS, explain_rows
    model = BAirOpsModel.load(bundle_path)
    X, _, _ = build_features(data, 24, 6)
    X = X.tail(20)
    ex, glob, err = explain_rows(model, X)
    assert err < 1e-3
    p = model.predict(X)
    for e, pi in zip(ex, p):
        assert abs(e["base_value"] + sum(d["effect"] for d in e["drivers"]) - pi) < 0.2
        assert len(e["drivers"]) == len(DRIVERS) and e["summary"].startswith("The forecast")
    assert {g["name"] for g in glob} == set(DRIVERS)


def test_explanations_real_learners(data):
    """Same check with the real libraries used in production (TreeSHAP paths)."""
    import pytest
    xgb = pytest.importorskip("xgboost")
    lgb = pytest.importorskip("lightgbm")
    cb = pytest.importorskip("catboost")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from bairops.explain import explain_rows
    X, y, _ = build_features(data, 24, 6)
    models = {"RandomForest": RandomForestRegressor(n_estimators=20, max_depth=8, random_state=0).fit(X, y),
              "XGBoost": xgb.XGBRegressor(n_estimators=40, max_depth=4).fit(X, y),
              "LightGBM": lgb.LGBMRegressor(n_estimators=40, num_leaves=15, verbose=-1).fit(X, y),
              "CatBoost": cb.CatBoostRegressor(iterations=40, depth=4, verbose=0).fit(X, y)}
    P = np.column_stack([m.predict(X) for m in models.values()])
    b = {"models": models, "model_order": list(models), "features": list(X.columns), "kind": "Ensemble-Stacking",
         "stacker": LinearRegression(positive=True).fit(P, y), "pi_halfwidth": 25.0, "horizon_h": 24}
    ex, glob, err = explain_rows(BAirOpsModel(b), X.tail(30))
    assert err < 1e-2, err
