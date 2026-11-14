"""Open-Meteo data for New Zealand: download (with a polite on-disk cache), solar position and PV power.

Time convention (important): Open-Meteo radiation at timestamp T is the MEAN over [T-1h, T); all other
variables are instantaneous at T. So the sun position that belongs to a row is the one at T - 30 min.
"""
import time

import numpy as np
import pandas as pd
import requests

from solarcast.nz import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
RENAME = dict(zip(config.HOURLY_VARS, config.COLUMNS))


# ---------- solar position and power ----------
def solar_position(times_utc, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Solar elevation and azimuth (degrees, azimuth clockwise from north) at UTC instants.

    NOAA general solar position formulas (fractional-year series for the equation of time and the declination),
    no atmospheric refraction. Accuracy is well below one degree, plenty for a clear-sky model.
    """
    t = pd.DatetimeIndex(times_utc)
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60 + t.second.to_numpy() / 3600
    g = 2 * np.pi / 365 * (t.dayofyear.to_numpy() - 1 + (hour - 12) / 24)          # fractional year (rad)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))     # minutes
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    solar_time = hour * 60 + eqtime + 4 * lon                                      # true solar time, minutes
    ha = np.radians(solar_time / 4 - 180)                                          # hour angle
    la = np.radians(lat)
    cos_zen = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    elevation = 90 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))
    azimuth = (180 + np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(decl) * np.cos(la)))) % 360
    return elevation, azimuth


def pv_power_kw(ghi, temp_c, system_kw: float = config.SYSTEM_KW) -> np.ndarray:
    """PV power from GHI: system_kw * ghi/1000 * PR * (1 + TEMP_COEFF * max(0, cell_temp - 25)), clipped.

    Cell temperature = air temperature + ghi/800 * 20 (NOCT-style). Derived, not measured.
    """
    ghi = np.asarray(ghi, dtype=float)
    cell = np.asarray(temp_c, dtype=float) + ghi / 800 * 20
    kw = system_kw * ghi / 1000 * config.PERFORMANCE_RATIO * (1 + config.TEMP_COEFF * np.maximum(0, cell - 25))
    return np.clip(kw, 0, system_kw)


def _to_frame(hourly: dict, region: str, lat: float, lon: float) -> pd.DataFrame:
    """One Open-Meteo `hourly` block -> tidy frame with solar position at the radiation interval midpoint."""
    df = pd.DataFrame(hourly).rename(columns=RENAME)
    df["time_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df = df.dropna(subset=["ghi_wm2"]).reset_index(drop=True)     # the newest days can still be empty
    df["solar_elevation_deg"], df["solar_azimuth_deg"] = solar_position(df["time_utc"] - pd.Timedelta("30min"), lat, lon)
    df.insert(0, "region", region)
    return df[["region", "time_utc"] + config.COLUMNS + ["solar_elevation_deg", "solar_azimuth_deg"]]


# ---------- download ----------
def _get(url: str, params: dict, tries: int = 6) -> dict:
    """GET with a patient retry (HTTP 429 rate limits, 5xx errors and timeouts)."""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=180)
        except (requests.Timeout, requests.ConnectionError):
            if k == tries - 1:
                raise
            time.sleep(20 * (k + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
            time.sleep(30 * (k + 1))
            continue
        r.raise_for_status()
    raise RuntimeError("unreachable")


def fetch_history(start: str, end: str, regions=config.REGIONS) -> pd.DataFrame:
    """Hourly history [start, end] (dates) for every region, long format. One request per region, cached as CSV."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in regions:
        lat, lon = config.REGIONS[name]
        path = config.RAW_DIR / f"{name.replace(' ', '_')}_{start}_{end}.csv"
        if path.exists():
            parts.append(pd.read_csv(path, parse_dates=["time_utc"]).assign(time_utc=lambda d: pd.to_datetime(d["time_utc"], utc=True)))
            continue
        params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                  "hourly": ",".join(config.HOURLY_VARS), "timezone": "UTC"}
        df = _to_frame(_get(ARCHIVE_URL, params)["hourly"], name, lat, lon)
        df.to_csv(path, index=False)
        parts.append(df)
        time.sleep(2)                                              # be polite
    return pd.concat(parts, ignore_index=True)


def fetch_live(regions=config.REGIONS) -> pd.DataFrame:
    """Last 48 h plus next 24 h for all regions in ONE request; `is_forecast` marks hours after the current hour."""
    names = list(regions)
    params = {"latitude": ",".join(str(config.REGIONS[n][0]) for n in names),
              "longitude": ",".join(str(config.REGIONS[n][1]) for n in names),
              "hourly": ",".join(config.HOURLY_VARS), "timezone": "UTC", "past_days": 2, "forecast_days": 2}
    data = _get(FORECAST_URL, params)
    data = data if isinstance(data, list) else [data]
    now = pd.Timestamp.now(tz="UTC").floor("h")
    parts = []
    for name, block in zip(names, data):
        df = _to_frame(block["hourly"], name, *config.REGIONS[name])
        df = df[(df["time_utc"] >= now - pd.Timedelta("48h")) & (df["time_utc"] <= now + pd.Timedelta("24h"))]
        parts.append(df.assign(is_forecast=df["time_utc"] > now))
    return pd.concat(parts, ignore_index=True)
