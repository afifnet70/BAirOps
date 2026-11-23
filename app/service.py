"""Serving logic (framework-independent so it can be unit-tested without a web server)."""
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bairops.cities import CITIES, CITY_BY_ID, CITY_BY_NAME  # noqa: E402
from bairops.config import PARAMS  # noqa: E402
from bairops.data import clean  # noqa: E402
from bairops.features import SEASON_NAMES, build_features  # noqa: E402
from bairops.metrics import aqi_label  # noqa: E402
from bairops.model import BAirOpsModel  # noqa: E402

REFRESH_MIN = int(os.environ.get("BAIROPS_REFRESH_MIN", "60"))
TZ = PARAMS["data"]["timezone_offset_h"]


def load_model():
    """Local bundle (BAIROPS_MODEL_PATH) for development/tests, otherwise the registry's production version."""
    local = os.environ.get("BAIROPS_MODEL_PATH")
    if local:
        return BAirOpsModel.load(local, version=os.environ.get("BAIROPS_MODEL_VERSION", "local"))
    from bairops import registry
    path, version, _ = registry.download_bundle(PARAMS["project"]["model_name"])
    return BAirOpsModel.load(path, version=version, source="MLflow model registry (production)")


class Service:
    def __init__(self, fetch_fn=None):
        if fetch_fn is None:
            from bairops.live import fetch_recent
            fetch_fn = fetch_recent
        self.fetch_fn = fetch_fn
        self.model = None
        self.forecasts, self.history = {}, {}
        self.last_refresh, self.last_error = None, None
        self.started = time.time()
        self._lock = threading.Lock()

    # ---- lifecycle --------------------------------------------------------------------------
    def start(self, background=True):
        if background:                                   # API comes up immediately; model + data load in the background
            threading.Thread(target=self._loop, daemon=True).start()
        else:
            self.model = load_model()
            self.refresh()

    def _loop(self):
        while True:
            if self.model is None:
                try:
                    self.model = load_model()
                    print(f"[model] loaded v{self.model.version} from {self.model.source}", flush=True)
                except Exception as e:                   # e.g. registry empty before the Bootstrap workflow ran
                    self.last_error = f"model not available yet: {type(e).__name__}: {e}"
                    print("[model]", self.last_error, flush=True)
                    time.sleep(120)
                    continue
            try:
                self.refresh()
            except Exception as e:                       # keep serving the last good forecasts
                self.last_error = f"{type(e).__name__}: {e}"
                print("[refresh] failed:", self.last_error, flush=True)
                time.sleep(300)
                continue
            time.sleep(REFRESH_MIN * 60)

    # ---- core -------------------------------------------------------------------------------
    def refresh(self):
        raw = self.fetch_fn()
        data = clean(raw, valid_start="2000-01-01", include_dhaka_2000_2022=True,
                     max_gap_h=PARAMS["data"]["max_interp_gap_h"])
        X, _, meta = build_features(data, self.model.horizon_h, TZ, for_inference=True)
        last = meta.groupby("city_id")["issue_time"].transform("max") == meta["issue_time"]
        Xl, ml = X[last.values], meta[last.values].reset_index(drop=True)
        t0 = time.perf_counter()
        yhat = self.model.predict(Xl)
        ms = (time.perf_counter() - t0) * 1000 / max(len(Xl), 1)
        lo, hi = self.model.interval(yhat)
        fc, hist = {}, {}
        for i, r in ml.iterrows():
            name = r["city_name"]
            info = CITY_BY_ID.get(int(r["city_id"]), {})
            fc[name.lower()] = {
                "city": name, "district": info.get("district"),
                "lat": info.get("lat"), "lon": info.get("lon"),
                "issued_at_utc": str(r["issue_time"]),
                "issued_at_local": str(r["issue_time"] + pd.Timedelta(hours=TZ)),
                "target_time_utc": str(r["target_time"]),
                "target_time_local": str(r["target_time"] + pd.Timedelta(hours=TZ)),
                "aqi_now": round(float(r["aqi_now"]), 1), "category_now": aqi_label(r["aqi_now"]),
                "aqi_forecast": round(float(yhat[i]), 1), "category_forecast": aqi_label(yhat[i]),
                "interval_90": [round(float(lo[i]), 1), round(float(hi[i]), 1)],
                "season": SEASON_NAMES[int(r["season"])], "model_version": self.model.version,
                "model_latency_ms": round(ms, 3)}
            h = data[data["city_id"] == r["city_id"]].tail(168)
            hist[name.lower()] = [{"t": str(t + pd.Timedelta(hours=TZ)), "aqi": (None if pd.isna(a) else round(float(a), 1))}
                                  for t, a in zip(h["datetime"], h["aqi"])]
        with self._lock:
            self.forecasts, self.history = fc, hist
            self.last_refresh, self.last_error = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), None
        print(f"[refresh] {len(fc)} cities updated, model v{self.model.version}", flush=True)

    # ---- queries ----------------------------------------------------------------------------
    def ready(self):
        return self.model is not None and bool(self.forecasts)

    def forecast(self, city):
        key = city.strip().lower()
        if key not in CITY_BY_NAME:
            raise KeyError(city)
        with self._lock:
            f = self.forecasts.get(key)
            h = self.history.get(key, [])
        if f is None:
            return None
        return dict(f, history_7d=h)

    def all_forecasts(self):
        with self._lock:
            return sorted(self.forecasts.values(), key=lambda f: -f["aqi_forecast"])

    def predict_rows(self, rows):
        X = pd.DataFrame(rows)
        missing = [c for c in self.model.features if c not in X.columns]
        if missing:
            raise ValueError(f"missing features: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        X = X[self.model.features].astype("float32")
        t0 = time.perf_counter()
        yhat = self.model.predict(X)
        ms = (time.perf_counter() - t0) * 1000
        lo, hi = self.model.interval(yhat)
        return {"model_version": self.model.version, "latency_ms": round(ms, 3),
                "predictions": [{"aqi_forecast": round(float(v), 1), "category": aqi_label(v),
                                 "interval_90": [round(float(a), 1), round(float(b), 1)]}
                                for v, a, b in zip(yhat, lo, hi)]}

    def health(self):
        status = "ok" if self.ready() else ("degraded" if self.last_error else "starting")
        return {"status": status,
                "model_loaded": self.model is not None, "model_version": getattr(self.model, "version", None),
                "cities_with_forecast": len(self.forecasts), "last_refresh_utc": self.last_refresh,
                "last_error": self.last_error, "uptime_s": round(time.time() - self.started, 1)}

    @staticmethod
    def cities():
        return [c["name"] for c in sorted(CITIES, key=lambda c: c["name"])]


def feature_template(model):
    """Example request body for POST /predict (all features set to typical values)."""
    return {"rows": [{f: 0.0 for f in model.features}]}


__all__ = ["Service", "load_model", "feature_template", "np"]
