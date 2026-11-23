"""HTTP-level tests (run in CI where FastAPI + httpx are installed)."""
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conftest import synthetic  # noqa: E402


@pytest.fixture()
def client(bundle_path, monkeypatch):
    monkeypatch.setenv("BAIROPS_MODEL_PATH", bundle_path)
    from fastapi.testclient import TestClient
    import app.main as main
    main.SERVICE.fetch_fn = lambda: synthetic(seed=2)
    with TestClient(main.app) as c:
        for _ in range(100):                      # background refresh runs in a thread
            if main.SERVICE.ready():
                break
            time.sleep(0.1)
        yield c


def test_endpoints(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model-info").json()["kind"] == "Ensemble-Stacking"
    from bairops.cities import CITIES
    cities = client.get("/cities").json()
    assert len(cities) == len(CITIES)
    allf = client.get("/forecast/all").json()
    assert len(allf) == 3
    r = client.get("/forecast", params={"city": allf[0]["city"]})
    assert r.status_code == 200 and "aqi_forecast" in r.json()
    assert client.get("/forecast", params={"city": "Atlantis"}).status_code == 404
    assert client.get("/").status_code == 200


def test_predict_endpoint(client):
    feats = client.get("/model-info").json()["features"]
    r = client.post("/predict", json={"rows": [{f: 1.0 for f in feats}]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 1
    assert client.post("/predict", json={"rows": [{"aqi": 1.0}]}).status_code == 422
