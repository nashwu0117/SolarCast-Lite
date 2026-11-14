"""Features for the one-hour-ahead GHI forecast.

Timing. Open-Meteo radiation stamped T is the mean over [T-1h, T). A row is an ISSUE time t: we stand at the end
of hour t and know every stamp <= t. The target is the radiation stamped t+1 = the mean over the NEXT hour [t, t+1).
Rows are indexed by (region, target_time = t+1h). Anything stamped t+1 or later is never used as an input.
Sun position and clear-sky values for the target are deterministic (astronomy), so they are allowed.

Day-ahead forecasts (v3/v4). The long frame can carry forecast columns fc_* (data.fetch_forecast_history) stamped at
the forecast's VALID time. The forecast for the target stamp t+1 is an allowed input even though t+1 is in the
future: it is Open-Meteo's `*_previous_day1` value, i.e. it was ISSUED about 24 hours earlier, long before t. It is the
ONLY thing stamped t+1 or later that is ever read. Forecasts for any other stamp are never used.
"""
import numpy as np
import pandas as pd

from solarcast.nz import config, data

HOUR = pd.Timedelta("1h")
HALF = pd.Timedelta("30min")
CS_MIN = 20.0           # W/m2: below this the clear-sky index is meaningless (dawn/dusk)

V1_FEATURES = [
    "ghi_t", "ghi_lag1", "ghi_lag2", "ghi_lag23", "ghi_roll3", "ghi_roll24", "d_ghi_1h",
    "kt_t", "kt_roll24", "cs_t",
    "dni_t", "dhi_t", "cloud_t", "d_cloud_1h", "temp_t", "d_temp_1h", "humidity_t", "d_humidity_1h",
    "wind_t", "precip_t",
    "elev_target", "azim_target", "cs_target", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "region_id", "lat",
]
FORECAST_FEATURES = ["fc_ghi_target", "fc_cloud_target", "fc_temp_target", "fc_dni_target", "fc_kt_target"]
FEATURES = V1_FEATURES + FORECAST_FEATURES      # v4 inputs; V1_FEATURES is the original list, kept for reference


def clear_sky_ghi(elevation_deg) -> np.ndarray:
    """Haurwitz clear-sky GHI (W/m2): 1098 * cos(z) * exp(-0.057 / cos(z)), zero when the sun is down."""
    cz = np.sin(np.radians(np.asarray(elevation_deg, dtype=float)))
    return np.where(cz > 0.01, 1098.0 * cz * np.exp(-0.057 / np.maximum(cz, 0.01)), 0.0)


def _location(df: pd.DataFrame, region: str) -> tuple[float, float, int]:
    """(lat, lon, region_id). A clicked map point (not one of the 14 regions) carries loc_lat/loc_lon/loc_region_id
    columns; its region_id is the nearest region's, so the model treats it like that region's climate."""
    if region in config.REGIONS:
        lat, lon = config.REGIONS[region]
        return lat, lon, list(config.REGIONS).index(region)
    return float(df["loc_lat"].iloc[0]), float(df["loc_lon"].iloc[0]), int(df["loc_region_id"].iloc[0])


def _region_features(df: pd.DataFrame, region: str) -> pd.DataFrame:
    lat, lon, region_id = _location(df, region)
    g = df.sort_values("time_utc").set_index("time_utc")
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC"))    # gaps become NaN
    ghi = g["ghi_wm2"]
    cs_t = pd.Series(clear_sky_ghi(g["solar_elevation_deg"]), index=g.index)
    kt = (ghi / cs_t).where(cs_t >= CS_MIN).clip(0, 1.5)
    f = pd.DataFrame(index=g.index)
    f["ghi_t"], f["ghi_lag1"], f["ghi_lag2"], f["ghi_lag23"] = ghi, ghi.shift(1), ghi.shift(2), ghi.shift(23)
    f["ghi_roll3"] = ghi.rolling(3, min_periods=2).mean()
    f["ghi_roll24"] = ghi.rolling(24, min_periods=12).mean()
    f["d_ghi_1h"] = ghi - ghi.shift(1)
    f["cs_t"], f["kt_t"] = cs_t, kt
    f["kt_roll24"] = (ghi.rolling(24, min_periods=12).sum() / cs_t.rolling(24, min_periods=12).sum().replace(0, np.nan)).clip(0, 1.5)
    f["dni_t"], f["dhi_t"], f["cloud_t"], f["temp_t"] = g["dni_wm2"], g["dhi_wm2"], g["cloud_cover_pct"], g["temp_c"]
    f["humidity_t"], f["wind_t"], f["precip_t"] = g["humidity_pct"], g["wind_kmh"], g["precip_mm"]
    f["d_cloud_1h"], f["d_temp_1h"] = f["cloud_t"] - f["cloud_t"].shift(1), f["temp_t"] - f["temp_t"].shift(1)
    f["d_humidity_1h"] = f["humidity_t"] - f["humidity_t"].shift(1)

    target_time = f.index + HOUR
    elev, azim = data.solar_position(target_time - HALF, lat, lon)       # midpoint of the target hour
    f["elev_target"], f["azim_target"], f["cs_target"] = elev, azim, clear_sky_ghi(elev)
    mid = target_time - HALF
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * mid.hour / 24), np.cos(2 * np.pi * mid.hour / 24)
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * mid.dayofyear / 365.25), np.cos(2 * np.pi * mid.dayofyear / 365.25)
    f["region_id"], f["lat"] = region_id, lat

    # Day-ahead forecast for the target stamp (issued ~24 h before t, see module docstring). NaN when unavailable.
    fc = {c: g[c] if c in g.columns else pd.Series(np.nan, index=g.index) for c in config.FORECAST_COLUMNS}
    f["fc_ghi_target"], f["fc_cloud_target"] = fc["fc_ghi_wm2"].shift(-1), fc["fc_cloud_pct"].shift(-1)
    f["fc_temp_target"], f["fc_dni_target"] = fc["fc_temp_c"].shift(-1), fc["fc_dni_wm2"].shift(-1)
    f["fc_kt_target"] = (f["fc_ghi_target"] / f["cs_target"]).where(f["cs_target"] >= CS_MIN).clip(0, 1.5)

    f["target_ghi"] = ghi.shift(-1)
    target_temp = g["temp_c"].shift(-1)
    f["target_kw"] = data.pv_power_kw(f["target_ghi"].fillna(0), target_temp.fillna(f["temp_t"]))
    f.loc[f["target_ghi"].isna(), "target_kw"] = np.nan
    f["meta_issue_time"], f["meta_cloud_now"] = f.index, f["cloud_t"]
    f["meta_target_temp"] = target_temp
    f["region"], f["target_time"] = region, target_time
    return f


def build_features(long_df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Feature table indexed by (region, target_time). Columns: FEATURES, target_ghi, target_kw, meta_*.

    require_target=False keeps the newest issue hour (target not yet known) so live forecasts can be made.
    """
    parts = [_region_features(g, r) for r, g in long_df.groupby("region", sort=False)]
    out = pd.concat(parts)
    if require_target:
        out = out[out["target_ghi"].notna()]
    out = out.set_index(["region", "target_time"]).sort_index()
    out["meta_elev_target"] = out["elev_target"]
    return out


def daytime_mask(feat: pd.DataFrame) -> pd.Series:
    """True where the target hour is daytime (sun above DAYTIME_ELEVATION_MIN at the hour's midpoint)."""
    return feat["meta_elev_target"] >= config.DAYTIME_ELEVATION_MIN
