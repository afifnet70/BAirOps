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
