"""Offline tests for data.py using a small CSV written in the real file format."""
import numpy as np
import pandas as pd

from solarcast import config, data

HEADER = (
    "datetime_utc,datetime_local,active_power_kw,capacity_factor,is_generating,"
    "temp_c,humidity_pct,cloud_cover_pct,wind_speed_kmh,precip_mm,ghi_wm2,dni_wm2,dhi_wm2,"
    "solar_elevation_deg,solar_azimuth_deg,inverter_temp_c"
)


def row(ts, power="0.0"):
    cf = "" if power == "" else str(float(power) / 50)
    gen = "" if power == "" else "1.0"
    weather = "" if power == "" else "10.0,60.0,40.0,5.0,0.0,300.0,200.0,100.0"
    if power == "":
        weather = ",,,,,,,"
    return f"{ts},{ts},{power},{cf},{gen},{weather},20.0,150.0,33.0"


def write_csv(tmp_path, monkeypatch):
    lines = [
        HEADER,
        row("2025-01-03 03:00:00", "5.0"),
        row("2025-01-03 01:00:00", "0.0"),   # out of order
        row("2025-01-03 02:00:00", "2.0"),
        row("2025-01-03 02:00:00", "2.0"),   # duplicate timestamp
        row("2025-01-03 05:00:00", ""),      # empty measurements (DST-style row)
        row("2025-01-03 06:00:00", "7.0"),   # 04:00 is missing entirely
    ]
    (tmp_path / "site_a_hourly_weather.csv").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # cached file, so nothing is downloaded


def test_load_site_schema_and_cleaning(tmp_path, monkeypatch):
    write_csv(tmp_path, monkeypatch)
    df = data.load_site("A")
    assert list(df.columns) == config.KEEP_COLS
    assert (df.dtypes == float).all()
    assert df.index.name == "datetime_utc" and str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert len(df) == 6  # 01:00 .. 06:00 complete grid
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    # real night zero kept, missing values stay NaN (never 0)
    assert df.loc["2025-01-03 01:00", config.TARGET_COL] == 0.0
    assert np.isnan(df.loc["2025-01-03 04:00", config.TARGET_COL])  # inserted gap row
    assert np.isnan(df.loc["2025-01-03 05:00", config.TARGET_COL])  # empty CSV row
    assert df.loc["2025-01-03 05:00", config.WEATHER_COLS].isna().all()


def test_quality_report(tmp_path, monkeypatch):
    write_csv(tmp_path, monkeypatch)
    rep = data.data_quality_report(data.load_site("A"))
    assert rep["n_rows"] == 6
    assert rep["n_duplicate_timestamps_in_csv"] == 1
    assert rep["n_inserted_gap_rows"] == 1
    assert rep["n_missing_power"] == 2
    assert rep["n_missing_weather"] == 2
    assert rep["start"] == pd.Timestamp("2025-01-03 01:00", tz="UTC")
    assert rep["end"] == pd.Timestamp("2025-01-03 06:00", tz="UTC")
    assert 0 <= rep["pct_daytime"] <= 100
    assert set(rep) == {"n_rows", "start", "end", "n_missing_power", "n_missing_weather",
                        "n_duplicate_timestamps_in_csv", "n_inserted_gap_rows", "pct_daytime"}


def test_download_uses_cache(tmp_path, monkeypatch):
    write_csv(tmp_path, monkeypatch)
    path = data.download_site_csv("A")  # would raise offline if it tried to download
    assert path == tmp_path / "site_a_hourly_weather.csv"
