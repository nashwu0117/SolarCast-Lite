import numpy as np
import pandas as pd

from solarcast.nz.features import FEATURES, build_features, clear_sky_ghi


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


def test_no_leakage_from_target_hour_onward(long_df):
    """Garbage in every row stamped >= target hour must not change that target row's features."""
    r = "Wellington"
    T = pd.Timestamp("2025-12-20 02:00", tz="UTC")
    base = build_features(long_df[long_df.region == r]).loc[(r, T), FEATURES]
    dirty = long_df[long_df.region == r].copy()
    cols = [c for c in dirty.columns if c not in ("region", "time_utc")]
    mask = dirty["time_utc"] >= T
    dirty.loc[mask, cols] = np.random.default_rng(1).uniform(-999, 999, (mask.sum(), len(cols)))
    changed = build_features(dirty, require_target=False).loc[(r, T), FEATURES]
    pd.testing.assert_series_equal(base, changed)


def test_require_target_false_keeps_newest_issue_hour(long_df):
    sub = long_df[long_df.region == "Auckland"]
    with_t, without = build_features(sub), build_features(sub, require_target=False)
    assert len(without) == len(with_t) + 1 and np.isnan(without["target_ghi"].iloc[-1])
