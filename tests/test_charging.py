import numpy as np
import pandas as pd
import pytest

from solarcast import charging

N = charging.WORK_END - charging.WORK_START


def test_ev_always_fully_charged_even_on_a_dark_day():
    out = charging.simulate_day(np.zeros(N), np.zeros(N))
    assert out["ev_kwh"] == pytest.approx(charging.EV_NEED_KWH)
    assert out["ev_grid_kwh"] == pytest.approx(charging.EV_NEED_KWH)  # no sun: all from the grid


def test_sunny_day_with_perfect_forecast_uses_only_solar():
    sun = np.full(N, 40.0)  # 40 kW all day: 34 kW surplus every hour
    out = charging.simulate_day(sun, sun)
    assert out["ev_grid_kwh"] == pytest.approx(0.0)
    assert out["ev_solar_kwh"] == pytest.approx(charging.EV_NEED_KWH)


def test_waiting_for_afternoon_sun_beats_charging_on_arrival():
    actual = np.r_[np.zeros(N // 2), np.full(N - N // 2, 30.0)]  # dark morning, sunny afternoon
    arrival = charging.simulate_day(charging.charge_on_arrival(N), actual)
    planned = charging.simulate_day(actual, actual)
    assert planned["ev_grid_kwh"] < arrival["ev_grid_kwh"]
    assert planned["ev_kwh"] == pytest.approx(charging.EV_NEED_KWH)


def test_simulate_summary_has_all_strategies():
    idx = pd.date_range("2026-03-02 00:00", periods=48, freq="h", tz="UTC", name="target_time")
    power = np.clip(30 * np.sin(np.pi * (idx.hour - 4) / 14), 0, None)
    feat = pd.DataFrame({"target": power}, index=idx)
    preds = {"Persistence": feat["target"].shift(1).fillna(0)}
    daily, summary = charging.simulate(feat, preds)
    assert set(summary.index) == {"Charge on arrival", "Persistence", "Perfect forecast"}
    assert (summary["days"] == 2).all()
    assert summary.loc["Perfect forecast", "ev_grid_kwh"] <= summary.loc["Charge on arrival", "ev_grid_kwh"]
