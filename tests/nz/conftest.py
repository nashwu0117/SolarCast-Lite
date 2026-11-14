"""Offline fixtures: synthetic NZ long frames shaped exactly like data.fetch_history output."""
import numpy as np
import pandas as pd
import pytest

from solarcast.nz import config, data
from solarcast.nz.features import clear_sky_ghi

TEST_REGIONS = ["Auckland", "Wellington", "Queenstown"]


def make_long(start="2025-05-01", end="2026-03-01", regions=TEST_REGIONS, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    parts = []
    for r in regions:
        lat, lon = config.REGIONS[r]
        elev, azim = data.solar_position(idx - pd.Timedelta("30min"), lat, lon)
        cloud = np.empty(len(idx))
        cloud[0] = 50
        for i in range(1, len(idx)):                       # slowly changing cloud (AR(1))
            cloud[i] = np.clip(cloud[i - 1] + 0.1 * (50 - cloud[i - 1]) + rng.normal(0, 12), 0, 100)
        ghi = clear_sky_ghi(elev) * (1 - 0.75 * (cloud / 100) ** 1.5)
        df = pd.DataFrame({"region": r, "time_utc": idx, "ghi_wm2": ghi, "dni_wm2": ghi * 0.6, "dhi_wm2": ghi * 0.3,
                           "cloud_cover_pct": cloud, "temp_c": 12 + 8 * np.sin(idx.day_of_year / 58) + rng.normal(0, 1, len(idx)),
                           "humidity_pct": rng.uniform(40, 95, len(idx)), "wind_kmh": rng.uniform(0, 30, len(idx)),
                           "precip_mm": 0.0, "is_day": (elev > 0).astype(float),
                           "solar_elevation_deg": elev, "solar_azimuth_deg": azim})
        # day-ahead forecast columns (valid-time stamped): the truth plus noise, like a decent forecast
        df["fc_ghi_wm2"] = np.clip(ghi + rng.normal(0, 40, len(idx)), 0, None)
        df["fc_cloud_pct"] = np.clip(cloud + rng.normal(0, 15, len(idx)), 0, 100)
        df["fc_temp_c"] = df["temp_c"] + rng.normal(0, 1.5, len(idx))
        df["fc_dni_wm2"] = np.clip(df["dni_wm2"] + rng.normal(0, 40, len(idx)), 0, None)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="session")
def long_df() -> pd.DataFrame:
    return make_long()
