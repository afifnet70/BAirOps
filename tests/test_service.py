import os

import pytest

from conftest import synthetic


@pytest.fixture()
def service(bundle_path, monkeypatch):
    monkeypatch.setenv("BAIROPS_MODEL_PATH", bundle_path)
    monkeypatch.setenv("BAIROPS_MODEL_VERSION", "3")
    from app.service import Service
    s = Service(fetch_fn=lambda: synthetic(seed=1))
    s.start(background=False)
    return s


def test_refresh_and_forecast(service):
    assert service.ready()
    h = service.health()
    assert h["status"] == "ok" and h["model_version"] == "3" and h["cities_with_forecast"] == 3
    name = service.all_forecasts()[0]["city"]
    f = service.forecast(name)
    for k in ("aqi_now", "aqi_forecast", "interval_90", "category_forecast", "target_time_local", "history_7d", "lat", "lon"):
        assert k in f
    assert len(f["history_7d"]) == 168
    assert f["interval_90"][0] <= f["aqi_forecast"] <= f["interval_90"][1]


def test_unknown_city(service):
    with pytest.raises(KeyError):
        service.forecast("Atlantis")


def test_predict_rows(service):
    row = {f: 1.0 for f in service.model.features}
    out = service.predict_rows([row, row])
    assert len(out["predictions"]) == 2 and out["model_version"] == "3"
    with pytest.raises(ValueError):
        service.predict_rows([{"aqi": 100.0}])
