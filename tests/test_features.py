"""Offline tests for features.py, including a leakage check."""
import numpy as np
import pandas as pd
import pytest

from solarcast import config
from solarcast.features import FEATURE_GROUPS, FEATURE_SETS, build_features

LEAK_COLS = [config.TARGET_COL, "capacity_factor", "is_generating"] + config.WEATHER_COLS


def test_feature_sets_match_contract():
    assert set(FEATURE_GROUPS) == {"power", "weather", "time_solar"}
    assert FEATURE_SETS["A"] == FEATURE_GROUPS["power"] + FEATURE_GROUPS["time_solar"]
    assert FEATURE_SETS["B"] == FEATURE_SETS["A"] + FEATURE_GROUPS["weather"]
    assert len(FEATURE_SETS["B"]) == len(set(FEATURE_SETS["B"]))


def test_columns_and_meta(site_df):
    feat = build_features(site_df)
    assert feat.index.name == "target_time"
    for col in FEATURE_SETS["B"] + ["target"]:
        assert col in feat.columns
    meta = [c for c in feat.columns if c.startswith("meta_")]
    assert set(meta) == {"meta_issue_time", "meta_target_elevation",
                         "meta_target_cloud_cover", "meta_target_capacity_factor"}
    for cols in FEATURE_SETS.values():
        assert not set(meta) & set(cols)
        assert "target" not in cols


def test_target_is_power_one_hour_later(site_df):
    feat = build_features(site_df)
    expected = site_df[config.TARGET_COL].reindex(feat.index)
    pd.testing.assert_series_equal(feat["target"], expected, check_names=False, check_freq=False)
    assert (feat["meta_issue_time"] == feat.index - pd.Timedelta("1h")).all()
    # lag features point backwards
    t = feat.index[100]
    issue = t - pd.Timedelta("1h")
    p = site_df[config.TARGET_COL]
    assert feat.loc[t, "power_lag1"] == p[issue]
    assert feat.loc[t, "power_lag6"] == p[issue - pd.Timedelta("5h")]
    assert feat.loc[t, "power_same_hour_yesterday"] == p[t - pd.Timedelta("24h")]
    assert np.isclose(feat.loc[t, "power_roll3_mean"], p[issue - pd.Timedelta("2h"):issue].mean())


def test_dropna_leaves_no_nan(site_df):
    feat = build_features(site_df)
    assert len(feat) > 0
    assert not feat[FEATURE_SETS["B"] + ["target"]].isna().any().any()
    assert len(build_features(site_df, dropna=False)) > len(feat)  # the DST hole is dropped


@pytest.mark.parametrize("target_time", ["2025-06-15 10:00", "2025-09-01 03:00", "2026-02-10 12:00"])
def test_no_leakage_from_target_hour_or_later(site_df, target_time):
    """Wrecking power/weather at rows >= T must not change any feature of target hour T."""
    T = pd.Timestamp(target_time, tz="UTC")
    base = build_features(site_df, dropna=False)
    wrecked_df = site_df.copy()
    wrecked_df.loc[wrecked_df.index >= T, LEAK_COLS] = 9999.0
    wrecked = build_features(wrecked_df, dropna=False)
    pd.testing.assert_series_equal(
        base.loc[T, FEATURE_SETS["B"]], wrecked.loc[T, FEATURE_SETS["B"]], check_names=False
    )
    # control: wrecking the issue hour (T-1h) too *should* change features, so the test can fail
    wrecked_df.loc[T - pd.Timedelta("1h"), LEAK_COLS] = 9999.0
    control = build_features(wrecked_df, dropna=False)
    assert control.loc[T, "power_lag1"] == 9999.0
