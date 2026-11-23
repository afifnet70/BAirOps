import numpy as np
import pandas as pd

from bairops.features import HISTORY_HOURS, bd_season, build_features


def test_shapes_and_target_alignment(data):
    X, y, meta = build_features(data, 24, 6)
    assert len(X) == len(y) == len(meta) > 0
    assert not X.isna().any().any()
    # target is the AQI exactly 24 h after the issue time of the same city
    r = meta.iloc[10]
    obs = data[(data.city_id == r.city_id) & (data.datetime == r.issue_time + pd.Timedelta(hours=24))]["aqi"].iloc[0]
    assert np.isclose(y.iloc[10], obs, atol=1e-3)


def test_no_future_leakage(data):
    """Changing the future must not change the features of earlier issue times."""
    X1, _, m1 = build_features(data, 24, 6, for_inference=True)
    d2 = data.copy()
    cut = d2.datetime.max() - pd.Timedelta(hours=30)
    d2.loc[d2.datetime > cut, ["aqi", "pm2_5", "pm10"]] *= 3
    X2, _, m2 = build_features(d2, 24, 6, for_inference=True)
    early = (m1.issue_time <= cut).values
    assert np.allclose(X1[early].values, X2[early].values, equal_nan=True)


def test_inference_keeps_latest_hour(data):
    _, _, m_train = build_features(data, 24, 6)
    _, _, m_inf = build_features(data, 24, 6, for_inference=True)
    assert m_inf.issue_time.max() == data.datetime.max()
    assert m_train.issue_time.max() == data.datetime.max() - pd.Timedelta(hours=24)
    # first usable row needs a full week of history
    first = m_inf.groupby("city_id").issue_time.min()
    assert (first >= data.datetime.min() + pd.Timedelta(hours=HISTORY_HOURS)).all()


def test_seasons():
    assert list(bd_season([1, 4, 7, 10, 12])) == [0, 1, 2, 3, 0]
