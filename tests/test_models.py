"""Model tests on a small hand-built feature frame (does not need features.py to exist)."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from solarcast import config, models

POWER = ["power_lag1", "power_lag2", "power_lag3", "power_lag6", "power_roll3_mean",
         "power_roll6_mean", "power_same_hour_yesterday"]
TIME_SOLAR = ["target_hour_sin", "target_hour_cos", "target_doy_sin", "target_doy_cos",
              "target_solar_elevation", "target_solar_azimuth", "issue_solar_elevation"]
WEATHER = config.WEATHER_COLS + ["cloud_cover_diff1", "ghi_diff1"]
LOCAL_SETS = {"A": POWER + TIME_SOLAR, "B": POWER + TIME_SOLAR + WEATHER}


@pytest.fixture
def sets(monkeypatch):
    """Feature sets from features.py if it exists, else the contract's column lists."""
    try:
        from solarcast.features import FEATURE_SETS
        return FEATURE_SETS
    except ImportError:
        stub = types.ModuleType("solarcast.features")
        stub.FEATURE_SETS = LOCAL_SETS
        monkeypatch.setitem(sys.modules, "solarcast.features", stub)
        return LOCAL_SETS


@pytest.fixture
def feat(sets):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-06-01", periods=300, freq="h", tz="UTC", name="target_time")
    df = pd.DataFrame(rng.random((300, len(sets["B"]))), index=idx, columns=sets["B"])
    df["target"] = df["power_lag1"] * 40 + rng.normal(0, 1, 300)
    for c in ["meta_target_elevation", "meta_target_cloud_cover", "meta_target_capacity_factor"]:
        df[c] = 1e6                 # junk: if a model used these it would show up
    df["meta_issue_time"] = idx - pd.Timedelta(hours=1)
    return df


def test_make_model_is_unfitted_and_deterministic():
    m = models.make_model()
    assert not hasattr(m, "n_iter_")
    assert m.early_stopping is False and m.random_state == config.RANDOM_STATE


def test_train_sets_feature_set_and_uses_only_its_columns(feat, sets):
    for fs in ("A", "B"):
        m = models.train_model(feat, fs)
        assert m.feature_set == fs
        assert m.n_features_in_ == len(sets[fs])
    # meta_* / target never used: wiping them must not change predictions
    m = models.train_model(feat, "A")
    changed = feat.copy()
    changed["target"] = 0.0
    changed[[c for c in feat if c.startswith("meta_") and c != "meta_issue_time"]] = -5.0
    pd.testing.assert_series_equal(models.predict(m, feat), models.predict(m, changed))


def test_predict_clips_to_capacity_range(feat):
    m = models.train_model(feat, "A")
    wild = feat.copy()
    wild["power_lag1"] = 1e6
    wild["target_solar_elevation"] = -1e6
    p = models.predict(m, wild)
    assert p.index.equals(feat.index)
    assert p.between(0, config.CAPACITY_KW).all()
    # a model that predicts huge/negative raw values is clipped
    train = feat.copy()
    train["target"] = 500.0
    assert (models.predict(models.train_model(train, "A"), feat) == config.CAPACITY_KW).all()
    train["target"] = -500.0
    assert (models.predict(models.train_model(train, "A"), feat) == 0).all()


def test_save_load_round_trip(feat, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    m = models.train_model(feat, "B")
    path = models.save_model(m, "m_b")
    assert path.exists() and path.parent == tmp_path / "models"
    m2 = models.load_model("m_b")
    assert m2.feature_set == "B"
    pd.testing.assert_series_equal(models.predict(m, feat), models.predict(m2, feat))


def test_permutation_importance_table(feat):
    m = models.train_model(feat, "A")
    t = models.permutation_importance_table(m, feat, n_repeats=2)
    assert list(t.columns) == ["feature", "importance_mean", "importance_std"]
    assert t["importance_mean"].is_monotonic_decreasing
    assert t["feature"].iloc[0] in ("power_lag1", "power_roll3_mean", "power_roll6_mean")
