import numpy as np
import pandas as pd

from solarcast.nz import config, data


def test_regions_are_new_zealand_only():
    assert len(config.REGIONS) == 14
    assert all(-48 < lat < -34 and 166 < lon < 179 for lat, lon in config.REGIONS.values())
    assert "romania" not in str(config.REGIONS).lower()


def test_solar_position_summer_noon_auckland():
    t = pd.date_range("2025-12-21", periods=24, freq="h", tz="UTC")
    elev, azim = data.solar_position(t, *config.REGIONS["Auckland"])
    assert 74 < elev.max() < 78                     # 90 - (36.85 - 23.44)
    assert elev.min() < -20                         # night
    noon = elev.argmax()
    assert azim[noon] < 20 or azim[noon] > 340      # the sun is in the north at noon in NZ


def test_pv_power_bounds_and_temperature():
    kw = data.pv_power_kw([0, 500, 1000, 5000], [10, 10, 10, 10])
    assert kw[0] == 0 and kw[1] < kw[2] <= config.SYSTEM_KW and kw[3] == config.SYSTEM_KW
    assert data.pv_power_kw(800, 35) < data.pv_power_kw(800, 5)     # hot panels produce less


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def _hourly(n=6):
    h = {"time": [f"2025-01-01T{i:02d}:00" for i in range(n)]}
    for v in config.HOURLY_VARS:
        h[v] = [100.0 * i for i in range(n)]
    h["shortwave_radiation"][-1] = None             # newest hour not available yet
    return {"hourly": h}


def test_fetch_history_parses_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(data.requests, "get", lambda url, params, timeout: calls.append(params) or FakeResponse(_hourly()))
    df = data.fetch_history("2025-01-01", "2025-01-01", regions=["Auckland", "Dunedin"])
    assert set(df["region"]) == {"Auckland", "Dunedin"} and len(calls) == 2
    assert {"ghi_wm2", "temp_c", "solar_elevation_deg", "solar_azimuth_deg"} <= set(df.columns)
    assert df["ghi_wm2"].notna().all() and len(df) == 10      # the empty last hour was dropped
    again = data.fetch_history("2025-01-01", "2025-01-01", regions=["Auckland", "Dunedin"])
    assert len(calls) == 2 and len(again) == len(df)           # second call came from the cache
