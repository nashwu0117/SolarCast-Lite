"""Offline tests for demo.predict_at using a fake ExperimentResult (no model training)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from solarcast import config

features = pytest.importorskip("solarcast.features")
from solarcast import demo  # noqa: E402


@pytest.fixture(scope="module")
def fake_result(site_df):
    feat = features.build_features(site_df)
    test = feat.loc["2026-03-01":"2026-06-30"]
    preds = {
        "Model B": test["target"] * 0.95,
        "Model A": test["target"] * 0.9,
        "Persistence": test["power_lag1"],
        "Previous-day same hour": test["power_same_hour_yesterday"],
    }
    return SimpleNamespace(
        site="A", df=site_df, feat=feat,
        splits={"test": test},
        predictions={"test": preds},
        metrics={"test": pd.DataFrame({"mae_kw": [1.0, 2.0]}, index=["Model B", "Model A"])},
    )


def _daytime_time(result):
    test = result.splits["test"]
    return test.index[(test["meta_target_elevation"] > config.DAYTIME_ELEVATION_MIN).to_numpy()][100]


def test_predict_at_keys_and_types(fake_result):
    t = _daytime_time(fake_result)
    out = demo.predict_at(fake_result, t)
    for key in ("model_b_kw", "model_a_kw", "persistence_kw", "previous_day_kw",
                "actual_kw", "abs_error_kw", "capacity_factor"):
        assert isinstance(out[key], float), key
    assert isinstance(out["recent_power"], pd.Series)
    assert len(out["recent_power"]) == 6
    assert out["recent_power"].index[-1] == t - pd.Timedelta(hours=1)
    assert set(out["weather_now"]) == set(config.WEATHER_COLS)
    assert out["abs_error_kw"] == pytest.approx(abs(out["model_b_kw"] - out["actual_kw"]))
    assert out["capacity_factor"] == pytest.approx(out["actual_kw"] / config.CAPACITY_KW)


def test_predict_at_accepts_string(fake_result):
    t = _daytime_time(fake_result)
    out = demo.predict_at(fake_result, str(t))
    assert out["target_time"] == t


def test_build_demo_returns_blocks(fake_result):
    gr = pytest.importorskip("gradio")
    app = demo.build_demo({"A": fake_result})
    assert isinstance(app, gr.Blocks)
