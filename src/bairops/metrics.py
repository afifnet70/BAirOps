"""Evaluation metrics (regression + US-AQI category accuracy), identical to Phase 1."""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, max_error, mean_absolute_error, mean_squared_error, r2_score

AQI_BINS = [-np.inf, 50, 100, 150, 200, 300, np.inf]
AQI_LABELS = ["Good", "Moderate", "Unhealthy for sensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"]


def aqi_cat(v):
    return pd.cut(pd.Series(np.asarray(v, dtype=float)), bins=AQI_BINS, labels=False).astype(int).values


def aqi_label(v):
    return AQI_LABELS[int(aqi_cat([v])[0])]


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ct, cp = aqi_cat(y_true), aqi_cat(y_pred)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
            "MAPE_%": float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1.0)) * 100),
            "MaxError": float(max_error(y_true, y_pred)),
            "Bias": float(np.mean(y_pred - y_true)),
            "Category_Acc_%": float(accuracy_score(ct, cp) * 100),
            "Category_MacroF1_%": float(f1_score(ct, cp, average="macro", labels=np.unique(ct)) * 100)}


def very_unhealthy_mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true > 200
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else float("nan")
