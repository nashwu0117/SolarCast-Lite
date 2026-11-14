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
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# caches written by the v3 experiment used these column names (forecast stamped at the valid time)
_LEGACY_FC = {"fc_ghi_target": "fc_ghi_wm2", "fc_cloud_target": "fc_cloud_pct", "fc_temp_target": "fc_temp_c",
              "fc_dni_target": "fc_dni_wm2"}
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
            time.sleep(20 * (k + 1) if tries > 2 else 3)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
            time.sleep(30 * (k + 1) if tries > 2 else 3)
            continue
        r.raise_for_status()
    raise RuntimeError("unreachable")


def _forecast_frame(hourly: dict) -> pd.DataFrame:
    """One Previous Runs `hourly` block -> frame [time_utc, fc_*] stamped at the forecast's valid time."""
    df = pd.DataFrame({"time_utc": pd.to_datetime(hourly["time"], utc=True),
                       **{v: hourly[k] for k, v in config.FORECAST_VARS.items()}})
    return df


def fetch_forecast_history(start: str, end: str, regions=config.REGIONS) -> pd.DataFrame:
    """Day-ahead forecasts (`*_previous_day1`, issued ~24 h before the hour) for [start, end], long format, cached as CSV.

    Columns: region, time_utc, fc_ghi_wm2, fc_cloud_pct, fc_temp_c, fc_dni_wm2. Radiation stamped T is the mean over
    [T-1h, T), exactly like the archive. The archive only exists from config.FORECAST_START, so `start` is clipped to it.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    start = max(start, config.FORECAST_START)
    parts = []
    for name in regions:
        lat, lon = config.REGIONS[name]
        path = config.RAW_DIR / f"fc_{name.replace(' ', '_')}_{start}_{end}.csv"
        if not path.exists():
            params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                      "hourly": ",".join(config.FORECAST_VARS), "timezone": "UTC"}
            _forecast_frame(_get(PREVIOUS_RUNS_URL, params)["hourly"]).to_csv(path, index=False)
            time.sleep(2)
        df = pd.read_csv(path).rename(columns=_LEGACY_FC)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        parts.append(df[["time_utc"] + config.FORECAST_COLUMNS].assign(region=name))
    return pd.concat(parts, ignore_index=True)[["region", "time_utc"] + config.FORECAST_COLUMNS]


def add_forecasts(long_df: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Left-join forecast columns onto the long frame at their valid time (rows without a forecast stay NaN)."""
    base = long_df.drop(columns=[c for c in config.FORECAST_COLUMNS if c in long_df.columns])
    return base.merge(fc, on=["region", "time_utc"], how="left")


def fetch_history(start: str, end: str, regions=config.REGIONS, forecasts: bool = True) -> pd.DataFrame:
    """Hourly history [start, end] (dates) for every region, long format. One request per region, cached as CSV.

    With forecasts=True the day-ahead forecast columns (fc_*) are added from the Previous Runs API (from 2024 on).
    """
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
    out = pd.concat(parts, ignore_index=True)
    if forecasts:
        out = add_forecasts(out, fetch_forecast_history(start, end, regions))
    return out


def fetch_point(lat: float, lon: float, name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Weather + day-ahead forecasts for ONE clicked map point, long format like fetch_history (no disk cache).

    With start/end (dates): the archive (replay). Without: live, the last 48 h plus the next 24 h with `is_forecast`.
    Only two quick tries per request so a click never hangs for minutes on a rate limit.
    """
    loc = {"latitude": round(lat, 4), "longitude": round(lon, 4), "timezone": "UTC"}
    if start:
        weather = _get(ARCHIVE_URL, {**loc, "start_date": start, "end_date": end, "hourly": ",".join(config.HOURLY_VARS)}, tries=2)
        fc_params = {**loc, "start_date": max(start, config.FORECAST_START), "end_date": end}
    else:
        span = {"past_days": 2, "forecast_days": 2}
        weather = _get(FORECAST_URL, {**loc, **span, "hourly": ",".join(config.HOURLY_VARS)}, tries=2)
        fc_params = {**loc, **span}
    df = _to_frame(weather["hourly"], name, lat, lon)
    try:
        if end and end < config.FORECAST_START:
            raise ValueError("no archived forecasts before FORECAST_START")
        fc = _forecast_frame(_get(PREVIOUS_RUNS_URL, {**fc_params, "hourly": ",".join(config.FORECAST_VARS)}, tries=2)["hourly"])
        df = df.merge(fc, on="time_utc", how="left")
    except Exception:                                              # forecasts are an extra input: NaN is handled by the model
        df = df.assign(**{c: np.nan for c in config.FORECAST_COLUMNS})
    if not start:
        now = pd.Timestamp.now(tz="UTC").floor("h")
        df = df[(df["time_utc"] >= now - pd.Timedelta("48h")) & (df["time_utc"] <= now + pd.Timedelta("24h"))]
        df = df.assign(is_forecast=df["time_utc"] > now)
    return df.reset_index(drop=True)


def fetch_points(points, names, start: str | None = None, end: str | None = None, batch: int = 50,
                 tries: int = 3) -> pd.DataFrame:
    """Weather + day-ahead forecasts for MANY points (the nationwide grid), long format like fetch_point.

    `points` is a list of (lat, lon), `names` their row names. Open-Meteo answers several locations in one request,
    so this sends one weather and one forecast request per `batch` points. With start/end: the archive (replay);
    without: live, the last 48 h plus the next 24 h with `is_forecast`.
    """
    points, names = list(points), list(names)
    parts = []
    for i in range(0, len(points), batch):
        chunk, cn = points[i:i + batch], names[i:i + batch]
        loc = {"latitude": ",".join(f"{p[0]:.4f}" for p in chunk),
               "longitude": ",".join(f"{p[1]:.4f}" for p in chunk), "timezone": "UTC"}
        if start:
            weather = _get(ARCHIVE_URL, {**loc, "start_date": start, "end_date": end,
                                         "hourly": ",".join(config.HOURLY_VARS)}, tries=tries)
            fc_params = {**loc, "start_date": max(start, config.FORECAST_START), "end_date": end}
        else:
            span = {"past_days": 2, "forecast_days": 2}
            weather = _get(FORECAST_URL, {**loc, **span, "hourly": ",".join(config.HOURLY_VARS)}, tries=tries)
            fc_params = {**loc, **span}
        weather = weather if isinstance(weather, list) else [weather]
        try:
            if end and end < config.FORECAST_START:
                raise ValueError("no archived forecasts before FORECAST_START")
            fcd = _get(PREVIOUS_RUNS_URL, {**fc_params, "hourly": ",".join(config.FORECAST_VARS)}, tries=tries)
            fcd = fcd if isinstance(fcd, list) else [fcd]
        except Exception:                                          # forecasts are an extra input: NaN is handled
            fcd = [None] * len(chunk)
        for name, (lat, lon), block, fblock in zip(cn, chunk, weather, fcd):
            df = _to_frame(block["hourly"], name, lat, lon)
            if fblock is not None:
                df = df.merge(_forecast_frame(fblock["hourly"]), on="time_utc", how="left")
            else:
                df = df.assign(**{c: np.nan for c in config.FORECAST_COLUMNS})
            parts.append(df)
        if i + batch < len(points):
            time.sleep(1)                                          # be polite between batches
    out = pd.concat(parts, ignore_index=True)
    if not start:
        now = pd.Timestamp.now(tz="UTC").floor("h")
        out = out[(out["time_utc"] >= now - pd.Timedelta("48h")) & (out["time_utc"] <= now + pd.Timedelta("24h"))]
        out = out.assign(is_forecast=out["time_utc"] > now)
    return out.reset_index(drop=True)


def fetch_live(regions=config.REGIONS) -> pd.DataFrame:
    """Last 48 h plus next 24 h for all regions; `is_forecast` marks hours after the current hour.

    Two requests for all regions: the weather (forecast API) and the day-ahead forecast columns fc_* (Previous Runs API,
    `*_previous_day1`, which also returns values for the coming hours because that run was issued ~24 h earlier).
    """
    names = list(regions)
    loc = {"latitude": ",".join(str(config.REGIONS[n][0]) for n in names),
           "longitude": ",".join(str(config.REGIONS[n][1]) for n in names), "timezone": "UTC", "past_days": 2, "forecast_days": 2}
    data = _get(FORECAST_URL, {**loc, "hourly": ",".join(config.HOURLY_VARS)})
    data = data if isinstance(data, list) else [data]
    try:
        fcd = _get(PREVIOUS_RUNS_URL, {**loc, "hourly": ",".join(config.FORECAST_VARS)})
        fcd = fcd if isinstance(fcd, list) else [fcd]
    except Exception:                                              # forecasts are an extra input: NaN is handled by the model
        fcd = [None] * len(names)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    parts = []
    for name, block, fblock in zip(names, data, fcd):
        df = _to_frame(block["hourly"], name, *config.REGIONS[name])
        if fblock is not None:
            df = df.merge(_forecast_frame(fblock["hourly"]), on="time_utc", how="left")
        else:
            df = df.assign(**{c: np.nan for c in config.FORECAST_COLUMNS})
        df = df[(df["time_utc"] >= now - pd.Timedelta("48h")) & (df["time_utc"] <= now + pd.Timedelta("24h"))]
        parts.append(df.assign(is_forecast=df["time_utc"] > now))
    return pd.concat(parts, ignore_index=True)
