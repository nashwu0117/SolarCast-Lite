import numpy as np
import pandas as pd
import pytest

from solarcast.nz import config
from solarcast.nz.features import FEATURES, FORECAST_FEATURES, V1_FEATURES, build_features, clear_sky_ghi


def test_clear_sky_zero_at_night():
    assert clear_sky_ghi([-5, 0])[0] == 0 and clear_sky_ghi([60])[0] > 800


def test_shape_and_target_is_next_hour(long_df):
    feat = build_features(long_df)
    assert feat.index.names == ["region", "target_time"]
    assert set(FEATURES) <= set(feat.columns) and {"target_ghi", "target_kw"} <= set(feat.columns)
    r = "Auckland"
    g = long_df[long_df.region == r].set_index("time_utc")["ghi_wm2"]
    T = pd.Timestamp("2025-12-20 02:00", tz="UTC")                     # target stamp: mean over [01:00, 02:00)
    row = feat.loc[(r, T)]
    assert row["target_ghi"] == g[T]
    assert row["ghi_t"] == g[T - pd.Timedelta("1h")]                   # latest known stamp is T-1
    assert row["ghi_lag1"] == g[T - pd.Timedelta("2h")]
    assert row["ghi_lag23"] == g[T - pd.Timedelta("24h")]


def _dirty_from(long_df, r, T, keep_forecast_at_T=True):
    """Garbage in every row stamped >= T. The forecast columns keep their value AT the target stamp T (allowed)."""
    dirty = long_df[long_df.region == r].copy()
    fc = config.FORECAST_COLUMNS
    cols = [c for c in dirty.columns if c not in ("region", "time_utc")]
    rng = np.random.default_rng(1)
    mask = (dirty["time_utc"] >= T).to_numpy()
    dirty.loc[mask, cols] = rng.uniform(-999, 999, (mask.sum(), len(cols)))
    if keep_forecast_at_T:
        at = (dirty["time_utc"] == T).to_numpy()
        dirty.loc[at, fc] = long_df[(long_df.region == r) & (long_df.time_utc == T)][fc].to_numpy()
    return dirty


def test_no_leakage_from_target_hour_onward(long_df):
    """Garbage in every row stamped >= target hour must not change that target row's features (v1 and v4 lists)."""
    r = "Wellington"
    T = pd.Timestamp("2025-12-20 02:00", tz="UTC")
    base = build_features(long_df[long_df.region == r]).loc[(r, T), FEATURES]
    # only the forecast for the target stamp T itself is exempt (it was issued ~24 h before)
    changed = build_features(_dirty_from(long_df, r, T), require_target=False).loc[(r, T), FEATURES]
    pd.testing.assert_series_equal(base, changed)
    assert list(FEATURES) == list(V1_FEATURES) + list(FORECAST_FEATURES)


def test_forecast_features_use_only_the_target_stamp(long_df):
    r, T = "Auckland", pd.Timestamp("2025-12-20 02:00", tz="UTC")
    sub = long_df[long_df.region == r]
    fc = sub.set_index("time_utc")
    row = build_features(sub).loc[(r, T)]
    assert row["fc_ghi_target"] == fc.loc[T, "fc_ghi_wm2"] and row["fc_cloud_target"] == fc.loc[T, "fc_cloud_pct"]
    assert row["fc_temp_target"] == fc.loc[T, "fc_temp_c"] and row["fc_dni_target"] == fc.loc[T, "fc_dni_wm2"]
    assert row["fc_kt_target"] == pytest.approx(min(1.5, row["fc_ghi_target"] / row["cs_target"]))
    # forecasts stamped at any OTHER time change nothing ...
    other = sub.copy()
    mask = other["time_utc"] != T
    other.loc[mask, config.FORECAST_COLUMNS] = 12345.0
    pd.testing.assert_series_equal(row[FEATURES].astype(float), build_features(other).loc[(r, T), FEATURES].astype(float))
    # ... while the forecast at the target stamp does feed the forecast features
    moved = sub.copy()
    moved.loc[moved["time_utc"] == T, "fc_ghi_wm2"] = 7.0
    assert build_features(moved).loc[(r, T), "fc_ghi_target"] == 7.0


def test_missing_forecasts_are_nan_not_errors(long_df):
    feat = build_features(long_df[long_df.region == "Auckland"].drop(columns=config.FORECAST_COLUMNS))
    assert feat[FORECAST_FEATURES].isna().all().all() and set(FEATURES) <= set(feat.columns)


def test_require_target_false_keeps_newest_issue_hour(long_df):
    sub = long_df[long_df.region == "Auckland"]
    with_t, without = build_features(sub), build_features(sub, require_target=False)
    assert len(without) == len(with_t) + 1 and np.isnan(without["target_ghi"].iloc[-1])
