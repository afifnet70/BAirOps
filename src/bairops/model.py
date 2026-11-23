"""Load a BAirOps model bundle (stacked ensemble) and predict on CPU."""
import joblib
import numpy as np


def _prepare_for_cpu_inference(name, m):
    """Models may have been trained on a GPU / with many threads; make single-request inference fast."""
    try:
        if name == "XGBoost":
            m.set_params(device="cpu", n_jobs=1)
            m.get_booster().set_param({"device": "cpu"})
        elif name == "RandomForest":
            m.set_params(n_jobs=1, verbose=0)        # avoids ~60 ms joblib start-up per request
        elif name == "LightGBM":
            m.set_params(n_jobs=1)
    except Exception as e:                            # never block serving because of a tuning flag
        print(f"[model] could not adjust {name}: {e}")
    return m


class BAirOpsModel:
    def __init__(self, bundle, version="local", source="local file"):
        self.bundle = bundle
        self.version = str(version)
        self.source = source
        self.features = list(bundle["features"])
        self.order = list(bundle.get("model_order", bundle["models"].keys()))
        for n in self.order:
            _prepare_for_cpu_inference(n, bundle["models"][n])
        self.kind = bundle.get("kind", "single")
        self.horizon_h = int(bundle.get("horizon_h", 24))
        self.pi_halfwidth = float(bundle.get("pi_halfwidth", np.nan))

    @classmethod
    def load(cls, path, version="local", source="local file"):
        return cls(joblib.load(path), version=version, source=source)

    def predict(self, X):
        Xs = X[self.features]
        P = np.column_stack([self.bundle["models"][n].predict(Xs) for n in self.order])
        if self.kind == "single":
            out = P[:, 0]
        elif self.kind == "Ensemble-Mean":
            out = P.mean(axis=1)
        elif self.kind == "Ensemble-Weighted":
            out = P @ np.asarray(self.bundle["weights"])
        else:                                          # Ensemble-Stacking (proposed model)
            out = self.bundle["stacker"].predict(P)
        return np.clip(np.asarray(out, float), 0, 500)

    def interval(self, yhat):
        q = self.pi_halfwidth if np.isfinite(self.pi_halfwidth) else 0.0
        yhat = np.asarray(yhat, float)
        return np.clip(yhat - q, 0, 500), np.clip(yhat + q, 0, 500)

    def info(self):
        b = self.bundle
        return {"model_name": "BAirOps stacked ensemble", "version": self.version, "source": self.source,
                "kind": self.kind, "base_models": self.order, "horizon_h": self.horizon_h,
                "n_features": len(self.features), "pi90_halfwidth": self.pi_halfwidth,
                "trained_on": b.get("trained_on"), "data_end": str(b.get("data_end", "")),
                "test_metrics": {k: (round(float(v), 4) if v == v else None)
                                 for k, v in (b.get("test_metrics") or {}).items()},
                "stacking_coef": [round(float(c), 4) for c in getattr(b.get("stacker"), "coef_", [])]}
