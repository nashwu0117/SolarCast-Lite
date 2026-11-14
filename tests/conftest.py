"""Shared offline test fixtures: a synthetic site frame shaped exactly like data.load_site output."""
import numpy as np
import pandas as pd
import pytest

from solarcast import config


def make_synthetic_site_df(start="2025-01-01", end="2026-07-01", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left", name="datetime_utc")
    hours = idx.hour.to_numpy() + 0.5
    doy = idx.dayofyear.to_numpy()
    # crude sun: peaks at 10:30 UTC (~12:30 local), higher in summer
    season = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    elev = np.clip(60 * season * np.cos(np.pi * (hours - 10.5) / 14), 0, None)
    cloud = np.clip(rng.normal(50, 30, len(idx)), 0, 100)
    ghi = 1000 * np.sin(np.radians(elev)) * (1 - 0.7 * cloud / 100)
    power = np.clip(ghi / 1000 * 50 * 0.85 + rng.normal(0, 1, len(idx)), 0, 50)
    power[elev == 0] = 0.0
    df = pd.DataFrame(index=idx)
    df["active_power_kw"] = power
    df["capacity_factor"] = power / 50.0
    df["is_generating"] = (power > 0).astype(float)
    df["temp_c"] = 10 + 10 * season + rng.normal(0, 2, len(idx))
    df["humidity_pct"] = np.clip(rng.normal(65, 15, len(idx)), 0, 100)
    df["cloud_cover_pct"] = cloud
    df["wind_speed_kmh"] = np.abs(rng.normal(8, 4, len(idx)))
    df["precip_mm"] = np.where(rng.random(len(idx)) < 0.05, rng.random(len(idx)) * 3, 0.0)
    df["ghi_wm2"] = ghi
    df["dni_wm2"] = ghi * 0.7
    df["dhi_wm2"] = ghi * 0.3
    df["solar_elevation_deg"] = elev
    df["solar_azimuth_deg"] = (hours * 15) % 360
    # simulate a DST-style hole: missing measurements, never zeros
    hole = (idx >= "2025-03-30 00:00") & (idx < "2025-03-30 03:00")
    df.loc[hole, ["active_power_kw", "capacity_factor", "is_generating"] + config.WEATHER_COLS] = np.nan
    return df[config.KEEP_COLS].astype(float)


@pytest.fixture(scope="session")
def site_df() -> pd.DataFrame:
    return make_synthetic_site_df()
