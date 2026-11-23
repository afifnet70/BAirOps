"""Training of the proposed stacked ensemble (same recipe as Phase 1, CPU-only for GitHub Actions).

Stage A : base learners on TRAIN, early stopping on VAL, stacking meta-learner fitted on VAL predictions
Stage B : base learners refitted on TRAIN+VAL (rounds x 1.1) and evaluated on the unseen HOLDOUT
Final   : (only if promoted) refit on ALL data (rounds x 1.2) -> production bundle
"""
import math
import platform
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .metrics import metrics

BASE = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]


def make_model(name, p, seed, max_rounds, n_rounds=None, early_stop=100):
    if name == "RandomForest":
        return RandomForestRegressor(**p, bootstrap=True, n_jobs=-1, random_state=seed)
    R = n_rounds or max_rounds
    if name == "XGBoost":
        import xgboost as xgb
        kw = {} if n_rounds else {"early_stopping_rounds": early_stop}
        return xgb.XGBRegressor(**p, n_estimators=R, tree_method="hist", objective="reg:squarederror",
                                eval_metric="rmse", n_jobs=-1, random_state=seed, verbosity=0, **kw)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**p, n_estimators=R, subsample_freq=1, objective="regression",
                                 n_jobs=-1, random_state=seed, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**p, iterations=R, loss_function="RMSE", bootstrap_type="Bernoulli",
                                 thread_count=-1, random_seed=seed, verbose=0, allow_writing_files=False)
    raise ValueError(name)


def fit(name, p, seed, max_rounds, early_stop, Xa, ya, Xv=None, yv=None, n_rounds=None):
    m = make_model(name, p, seed, max_rounds, n_rounds, early_stop)
    es = Xv is not None and n_rounds is None and name != "RandomForest"
    if name == "XGBoost" and es:
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        return m, int(m.best_iteration) + 1
    if name == "LightGBM" and es:
        import lightgbm as lgb
        m.fit(Xa, ya, eval_set=[(Xv, yv)], eval_metric="rmse", callbacks=[lgb.early_stopping(early_stop, verbose=False)])
        bi = m.best_iteration_
        return m, int(bi) if bi and bi > 0 else max_rounds
    if name == "CatBoost" and es:
        m.fit(Xa, ya, eval_set=(Xv, yv), early_stopping_rounds=early_stop, use_best_model=True)
        return m, int(m.get_best_iteration()) + 1
    m.fit(Xa, ya)
    return m, n_rounds or p.get("n_estimators")


def _versions():
    import catboost, lightgbm, sklearn, xgboost
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__}


def train_candidate(X, y, meta, params, log=print):
    """Returns dict with stage-B models, holdout predictions/metrics, rounds, stacker, PI half-width."""
    tp, seed = params["train"], params["project"]["seed"]
    end = meta["target_time"].max()
    hold_start = end - pd.Timedelta(days=tp["holdout_days"])
    val_start = hold_start - pd.Timedelta(days=tp["val_days"])
    tt = meta["target_time"]
    tr, va, ho = (tt < val_start).values, ((tt >= val_start) & (tt < hold_start)).values, (tt >= hold_start).values
    log(f"split: train {tr.sum():,} | val {va.sum():,} ({val_start:%Y-%m-%d}) | holdout {ho.sum():,} ({hold_start:%Y-%m-%d} .. {end:%Y-%m-%d})")
    Xtr, ytr, Xva, yva, Xho, yho = X[tr], y[tr].values, X[va], y[va].values, X[ho], y[ho].values

    rounds, val_pred = {}, {}
    for n in tp["base_models"]:
        t0 = time.time()
        m, r = fit(n, tp["params"][n], seed, tp["max_boost_rounds"], tp["early_stop"], Xtr, ytr, Xva, yva)
        rounds[n], val_pred[n] = r, m.predict(Xva)
        log(f"stage A {n:12s} rounds={r:<5} val RMSE={metrics(yva, val_pred[n])['RMSE']:.3f} ({time.time()-t0:.0f}s)")
        del m
    P_va = np.column_stack([val_pred[n] for n in tp["base_models"]])
    stacker = LinearRegression(positive=True).fit(P_va, yva)
    r_val = np.abs(yva - stacker.predict(P_va))
    alpha = params.get("pi_alpha", 0.1)
    q = float(np.quantile(r_val, min(1.0, math.ceil((len(r_val) + 1) * (1 - alpha)) / len(r_val))))
    log(f"stacking coef={np.round(stacker.coef_, 3).tolist()} | val RMSE={metrics(yva, stacker.predict(P_va))['RMSE']:.3f}")

    models, ho_pred = {}, {}
    Xtv, ytv = pd.concat([Xtr, Xva]), np.concatenate([ytr, yva])
    for n in tp["base_models"]:
        t0 = time.time()
        nr = None if n == "RandomForest" else int(math.ceil(rounds[n] * 1.1))
        models[n], _ = fit(n, tp["params"][n], seed, tp["max_boost_rounds"], tp["early_stop"], Xtv, ytv, n_rounds=nr)
        ho_pred[n] = models[n].predict(Xho)
        log(f"stage B {n:12s} holdout RMSE={metrics(yho, ho_pred[n])['RMSE']:.3f} ({time.time()-t0:.0f}s)")
    P_ho = np.column_stack([ho_pred[n] for n in tp["base_models"]])
    ens = np.clip(stacker.predict(P_ho), 0, 500)
    return {"models": models, "rounds": rounds, "stacker": stacker, "pi_halfwidth": q,
            "holdout_mask": ho, "holdout_pred": ens, "holdout_true": yho,
            "holdout_metrics": metrics(yho, ens), "single_holdout_metrics": {n: metrics(yho, ho_pred[n]) for n in ho_pred},
            "hold_start": hold_start, "val_start": val_start, "end": end}


def refit_all(X, y, cand, params, log=print):
    tp, seed = params["train"], params["project"]["seed"]
    out = {}
    for n in tp["base_models"]:
        t0 = time.time()
        nr = None if n == "RandomForest" else int(math.ceil(cand["rounds"][n] * 1.2))
        out[n], _ = fit(n, tp["params"][n], seed, tp["max_boost_rounds"], tp["early_stop"], X, y.values, n_rounds=nr)
        log(f"final   {n:12s} refit on all {len(X):,} rows ({time.time()-t0:.0f}s)")
    return out


def make_bundle(models, features, cand, params, trained_on):
    return {"models": models, "model_order": list(params["train"]["base_models"]), "features": list(features),
            "base_models": list(params["train"]["base_models"]), "selected": "Ensemble-Stacking",
            "kind": "Ensemble-Stacking", "stacker": cand["stacker"], "weights": None,
            "pi_halfwidth": cand["pi_halfwidth"], "horizon_h": params["features"]["horizon_h"],
            "trained_on": trained_on, "data_end": str(cand["end"]), "test_metrics": cand["holdout_metrics"],
            "rounds": cand["rounds"], "params": params["train"]["params"], "versions": _versions()}
