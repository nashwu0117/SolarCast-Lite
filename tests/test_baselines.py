"""Offline tests for baselines.py."""
import pandas as pd

from solarcast.baselines import BASELINES, persistence, previous_day_same_hour
from solarcast.features import build_features


def test_baselines_return_right_columns(site_df):
    feat = build_features(site_df)
    assert set(BASELINES) == {"Persistence", "Previous-day same hour"}
    p = persistence(feat)
    d = previous_day_same_hour(feat)
    assert isinstance(p, pd.Series) and p.index.equals(feat.index)
    assert (p == feat["power_lag1"]).all()
    assert (d == feat["power_same_hour_yesterday"]).all()
    for fn in BASELINES.values():
        assert not fn(feat).isna().any()
