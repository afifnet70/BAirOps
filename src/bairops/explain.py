"""Explainable AI for the BAirOps stacked ensemble.

For every forecast we split the predicted AQI into a base value plus one contribution per input
feature, so that  base + sum(contributions) = forecast.

* XGBoost, LightGBM and CatBoost: exact TreeSHAP values from each library's own implementation.
* Random Forest (and any sklearn tree): path decomposition (Saabas), which is exact and additive
  for every tree and therefore for the forest average.
* Linear models (only used by the test fixtures): coef * x.

The stacked ensemble is a linear combination of the four learners,
    y = b0 + sum_k beta_k * y_k,
so its contributions are sum_k beta_k * phi_k and its base value is b0 + sum_k beta_k * base_k.
The 83 features are then grouped into a few drivers that people can read.
"""
import re
import warnings

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------------------------
# per-learner contributions
# ---------------------------------------------------------------------------------------------
def _tree_path_contrib(est, X):
    """Saabas path contributions for one fitted sklearn regression tree."""
    t = est.tree_
    n_nodes, n_feat = t.node_count, X.shape[1]
    val = t.value[:, 0, 0].astype(float)
    parent = np.full(n_nodes, -1)
    for side in (t.children_left, t.children_right):
        ok = side >= 0
        parent[side[ok]] = np.nonzero(ok)[0]
    child = np.nonzero(parent >= 0)[0]
    M = sp.csr_matrix((val[child] - val[parent[child]], (child, t.feature[parent[child]])), shape=(n_nodes, n_feat))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)     # numpy input to a model fitted on a DataFrame
        dp = est.decision_path(X)
    return np.asarray((dp @ M).todense()), val[0]


def _forest_contrib(m, X):
    ests = getattr(m, "estimators_", None)
    if ests is None:                                  # a single tree
        c, b = _tree_path_contrib(m, X)
        return c, np.full(len(X), b)
    C, B = np.zeros(X.shape), 0.0
    for e in ests:
        c, b = _tree_path_contrib(e, X)
        C += c
        B += b
    return C / len(ests), np.full(len(X), B / len(ests))


def learner_contrib(m, Xdf):
    """Return (contributions n x F, base n) for one fitted learner."""
    mod = type(m).__module__
    X = Xdf.to_numpy(dtype=np.float32)
    if mod.startswith("xgboost"):
        import xgboost as xgb
        kw = {}
        bi = getattr(m, "best_iteration", None)
        if bi is not None:
            kw["iteration_range"] = (0, int(bi) + 1)
        out = m.get_booster().predict(xgb.DMatrix(Xdf), pred_contribs=True, **kw)
        return out[:, :-1], out[:, -1]
    if mod.startswith("lightgbm"):
        out = np.asarray(m.predict(Xdf, pred_contrib=True))
        return out[:, :-1], out[:, -1]
    if mod.startswith("catboost"):
        from catboost import Pool
        out = np.asarray(m.get_feature_importance(data=Pool(Xdf), type="ShapValues"))
        return out[:, :-1], out[:, -1]
    if hasattr(m, "tree_") or hasattr(m, "estimators_"):
        return _forest_contrib(m, X)
    if hasattr(m, "coef_"):                           # linear stand-ins (tests)
        coef = np.ravel(m.coef_)
        return X * coef, np.full(len(X), float(np.ravel([m.intercept_])[0]))
    raise TypeError(f"no explainer for {type(m)}")


def contributions(model, X):
    """model: BAirOpsModel. Returns (contrib n x F, base n, raw_prediction n)."""
    Xs = X[model.features]
    b = model.bundle
    parts = [learner_contrib(b["models"][n], Xs) for n in model.order]
    if model.kind == "single":
        w, b0 = np.array([1.0]), 0.0
    elif model.kind == "Ensemble-Mean":
        w, b0 = np.full(len(parts), 1 / len(parts)), 0.0
    elif model.kind == "Ensemble-Weighted":
        w, b0 = np.asarray(b["weights"], float), 0.0
    else:
        w, b0 = np.ravel(b["stacker"].coef_).astype(float), float(np.ravel([b["stacker"].intercept_])[0])
    C = sum(wk * c for wk, (c, _) in zip(w, parts))
    base = b0 + sum(wk * bb for wk, (_, bb) in zip(w, parts))
    return C, base, base + C.sum(axis=1)


# ---------------------------------------------------------------------------------------------
# readable names and driver groups
# ---------------------------------------------------------------------------------------------
_POL = {"aqi": "AQI", "pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2", "ozone": "O3",
        "carbon_monoxide": "CO", "sulphur_dioxide": "SO2"}

DRIVERS = ["Recent AQI (last 12 h)", "Recent PM2.5 (last 12 h)", "Recent PM10 dust (last 12 h)", "Pollution 1-3 days ago",
           "Pollution over the past week", "Gases (NO2, O3, CO, SO2)", "Time of year", "Hour and weekday",
           "Location"]


def feature_label(f):
    m = re.match(r"(aqi|pm2_5|pm10|nitrogen_dioxide|ozone|carbon_monoxide|sulphur_dioxide)(?:_(lag|rmean|rstd)(\d+))?$", f)
    if m:
        p, kind, h = _POL[m.group(1)], m.group(2), m.group(3)
        if kind is None:
            return f"{p} now"
        if kind == "lag":
            return f"{p} {h} h ago"
        if kind == "rmean":
            return f"{p}, mean of last {h} h"
        return f"{p}, variability over {h} h"
    return {"aqi_diff1": "AQI change in the last hour", "aqi_diff24": "AQI change since yesterday",
            "pm_ratio": "PM2.5 / PM10 ratio", "lat": "Latitude", "lon": "Longitude",
            "hour_sin": "Hour of day", "hour_cos": "Hour of day", "hour": "Hour of day",
            "doy_sin": "Day of year", "doy_cos": "Day of year", "month": "Month", "season": "Season",
            "dayofweek": "Day of week", "is_weekend_bd": "Weekend (Fri-Sat)"}.get(f, f)


def feature_group(f):
    if f in ("lat", "lon"):
        return "Location"
    if f in ("doy_sin", "doy_cos", "month", "season"):
        return "Time of year"
    if f in ("hour_sin", "hour_cos", "hour", "dayofweek", "is_weekend_bd"):
        return "Hour and weekday"
    if re.match(r"(nitrogen_dioxide|ozone|carbon_monoxide|sulphur_dioxide)", f):
        return "Gases (NO2, O3, CO, SO2)"
    if f == "aqi_diff24":
        return "Pollution 1-3 days ago"
    m = re.search(r"_(?:lag|rmean|rstd)(\d+)$", f)
    h = int(m.group(1)) if m else 0
    if h >= 168:
        return "Pollution over the past week"
    if h >= 24:
        return "Pollution 1-3 days ago"
    if f.startswith("aqi"):
        return "Recent AQI (last 12 h)"
    if f.startswith("pm2_5"):
        return "Recent PM2.5 (last 12 h)"
    return "Recent PM10 dust (last 12 h)"        # pm10_* and pm_ratio


def _low(name):
    name = name.split(" (")[0]
    return name if name[:2].isupper() else name[0].lower() + name[1:]


def _fmt(v):
    return f"{'+' if v >= 0 else '\u2212'}{abs(v):.0f}"


def summarise(drivers, base, forecast):
    """One plain-language sentence for the home screen."""
    up = [d for d in drivers if d["effect"] >= 1][:2]
    down = [d for d in drivers if d["effect"] <= -1][:1]
    diff = forecast - base
    s = (f"The forecast ({forecast:.0f}) is {abs(diff):.0f} {'above' if diff >= 0 else 'below'} "
         f"the model's typical level ({base:.0f})")
    lead = up if diff >= 0 else [d for d in drivers if d["effect"] <= -1][:2]
    if lead:
        s += ", mainly because of " + " and ".join(f"{_low(d['name'])} ({_fmt(d['effect'])})" for d in lead)
    other = down if diff >= 0 else up[:1]
    if other:
        nm = other[0]["name"].split(" (")[0]
        s += f". {nm} pulls it {'down' if other[0]['effect'] < 0 else 'up'} ({_fmt(other[0]['effect'])})"
    return s + "."


def explain_rows(model, X, top_k=6):
    """Per-row explanation dicts (same order as X) and the global mean |effect| per driver."""
    C, base, raw = contributions(model, X)
    feats = model.features
    groups = np.array([feature_group(f) for f in feats])
    G = np.column_stack([C[:, groups == g].sum(axis=1) for g in DRIVERS])
    out = []
    for i in range(len(X)):
        drivers = sorted(({"name": g, "effect": round(float(G[i, j]), 2)} for j, g in enumerate(DRIVERS)),
                         key=lambda d: -abs(d["effect"]))
        order = np.argsort(-np.abs(C[i]))[:top_k]
        top = [{"feature": feats[j], "label": feature_label(feats[j]),
                "value": round(float(X[feats[j]].iloc[i]), 2), "effect": round(float(C[i, j]), 2)} for j in order]
        fc = float(np.clip(raw[i], 0, 500))
        out.append({"method": "TreeSHAP (XGBoost, LightGBM, CatBoost) and path decomposition (Random Forest), "
                              "combined with the stacking weights",
                    "base_value": round(float(base[i]), 1), "forecast": round(fc, 1),
                    "drivers": drivers, "top_features": top, "summary": summarise(drivers, float(base[i]), fc)})
    glob = sorted(({"name": g, "mean_abs_effect": round(float(np.abs(G[:, j]).mean()), 2)} for j, g in enumerate(DRIVERS)),
                  key=lambda d: -d["mean_abs_effect"])
    return out, glob, float(np.max(np.abs(raw - model.predict_raw(X)))) if hasattr(model, "predict_raw") else None
