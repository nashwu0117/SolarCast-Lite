import numpy as np
import pandas as pd

from solarcast.nz import config, data


def test_regions_are_new_zealand_only():
    assert len(config.REGIONS) == 14
    assert all(-48 < lat < -34 and 166 < lon < 179 for lat, lon in config.REGIONS.values())


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
    df = data.fetch_history("2025-01-01", "2025-01-01", regions=["Auckland", "Dunedin"], forecasts=False)
    assert set(df["region"]) == {"Auckland", "Dunedin"} and len(calls) == 2
    assert {"ghi_wm2", "temp_c", "solar_elevation_deg", "solar_azimuth_deg"} <= set(df.columns)
    assert df["ghi_wm2"].notna().all() and len(df) == 10      # the empty last hour was dropped
    again = data.fetch_history("2025-01-01", "2025-01-01", regions=["Auckland", "Dunedin"], forecasts=False)
    assert len(calls) == 2 and len(again) == len(df)           # second call came from the cache


def _fc_hourly(n=6):
    h = {"time": [f"2025-01-01T{i:02d}:00" for i in range(n)]}
    for k in config.FORECAST_VARS:
        h[k] = [10.0 * i for i in range(n)]
    return {"hourly": h}


def test_forecast_history_columns_cache_and_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(data.requests, "get", lambda url, params, timeout: calls.append((url, params)) or FakeResponse(_fc_hourly()))
    fc = data.fetch_forecast_history("2020-01-01", "2025-01-01", regions=["Auckland"])
    url, params = calls[0]
    assert url == data.PREVIOUS_RUNS_URL and params["start_date"] == config.FORECAST_START      # clipped to 2024
    assert all(v.endswith("_previous_day1") for v in params["hourly"].split(","))
    assert list(fc.columns) == ["region", "time_utc"] + config.FORECAST_COLUMNS and len(fc) == 6
    data.fetch_forecast_history("2020-01-01", "2025-01-01", regions=["Auckland"])
    assert len(calls) == 1                                                                      # cached
    long = pd.DataFrame({"region": "Auckland", "time_utc": pd.date_range("2024-12-31 22:00", periods=6, freq="h", tz="UTC"),
                         "ghi_wm2": 1.0})
    out = data.add_forecasts(long, fc)
    assert len(out) == 6 and out["fc_ghi_wm2"].isna().sum() == 2 and out["fc_ghi_wm2"].iloc[2] == 0.0   # stamped at valid time


def test_fetch_live_has_forecast_columns_for_next_hour(monkeypatch):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    times = [(now + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(-5, 30)]

    def block(vars_):
        return {"hourly": {"time": times, **{v: [100.0] * len(times) for v in vars_}}}

    def fake_get(url, params, tries=6, timeout=180):
        vars_ = params["hourly"].split(",")
        return [block(vars_)] if url == data.FORECAST_URL else [block(vars_)]

    monkeypatch.setattr(data, "_get", fake_get)
    live = data.fetch_live(["Auckland"])
    assert set(config.FORECAST_COLUMNS) <= set(live.columns)
    nxt = live[live["time_utc"] == now + pd.Timedelta("1h")]
    assert len(nxt) == 1 and nxt["fc_ghi_wm2"].notna().all() and bool(nxt["is_forecast"].iloc[0])


def test_cut_off_reply_is_retried(monkeypatch):
    """A 200 reply with an empty or cut-off body is retried like a timeout."""
    from solarcast.nz import data as nzdata

    class Reply:
        status_code = 200
        def __init__(self, ok):
            self.ok = ok
        def json(self):
            if not self.ok:
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return {"hourly": {}}

    replies = iter([Reply(False), Reply(True)])
    monkeypatch.setattr(nzdata.requests, "get", lambda *a, **k: next(replies))
    monkeypatch.setattr(nzdata.time, "sleep", lambda s: None)
    assert nzdata._get("https://example.invalid", {}, tries=2, timeout=1) == {"hourly": {}}
