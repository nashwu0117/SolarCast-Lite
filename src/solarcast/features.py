"""Turn a load_site table into one row per target hour: features known at issue time + the target."""
import numpy as np
import pandas as pd

from solarcast import config

FEATURE_GROUPS = {
    "power": [
        "power_lag1", "power_lag2", "power_lag3", "power_lag6",
        "power_roll3_mean", "power_roll6_mean", "power_same_hour_yesterday",
    ],
    "weather": config.WEATHER_COLS + ["cloud_cover_diff1", "ghi_diff1"],
    "time_solar": [
        "target_hour_sin", "target_hour_cos", "target_doy_sin", "target_doy_cos",
        "target_solar_elevation", "target_solar_azimuth", "issue_solar_elevation",
    ],
}
FEATURE_SETS = {
    "A": FEATURE_GROUPS["power"] + FEATURE_GROUPS["time_solar"],  # power + sun only
    "B": FEATURE_GROUPS["power"] + FEATURE_GROUPS["time_solar"] + FEATURE_GROUPS["weather"],
}


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Build the model table. Row t of `df` is the issue hour; the target is power at t+1.

    Every feature uses rows <= t, except the target hour's sun position and calendar,
    which are fixed by astronomy and can be known in advance.
    """
    power = df[config.TARGET_COL]
    out = pd.DataFrame(index=df.index)  # indexed by issue time for now

    # Power history (shift(k) looks k hours back, so shifts are never negative here).
    out["power_lag1"] = power
    out["power_lag2"] = power.shift(1)
    out["power_lag3"] = power.shift(2)
    out["power_lag6"] = power.shift(5)
    out["power_roll3_mean"] = power.rolling(3).mean()
    out["power_roll6_mean"] = power.rolling(6).mean()
    out["power_same_hour_yesterday"] = power.shift(23)  # = power at (t+1) - 24h

    # Weather at issue hour t, plus how much cloud / sunlight changed in the last hour.
    for col in config.WEATHER_COLS:
        out[col] = df[col]
    out["cloud_cover_diff1"] = df["cloud_cover_pct"].diff()
    out["ghi_diff1"] = df["ghi_wm2"].diff()

    # Target-hour calendar and sun position (the only values taken from row t+1; deterministic).
    target_time = df.index + pd.Timedelta(hours=1)
    local = target_time.tz_convert(config.LOCAL_TZ)
    hour_angle = 2 * np.pi * local.hour.to_numpy() / 24
    doy_angle = 2 * np.pi * local.day_of_year.to_numpy() / 365.25
    out["target_hour_sin"] = np.sin(hour_angle)
    out["target_hour_cos"] = np.cos(hour_angle)
    out["target_doy_sin"] = np.sin(doy_angle)
    out["target_doy_cos"] = np.cos(doy_angle)
    out["target_solar_elevation"] = df["solar_elevation_deg"].shift(-1)
    out["target_solar_azimuth"] = df["solar_azimuth_deg"].shift(-1)
    out["issue_solar_elevation"] = df["solar_elevation_deg"]

    out["target"] = power.shift(-1)

    # Metadata for evaluation only; never model inputs.
    out["meta_issue_time"] = df.index
    out["meta_target_elevation"] = out["target_solar_elevation"]
    out["meta_target_cloud_cover"] = df["cloud_cover_pct"].shift(-1)
    out["meta_target_capacity_factor"] = df["capacity_factor"].shift(-1)

    out.index = pd.DatetimeIndex(target_time, name="target_time")
    if dropna:
        out = out.dropna(subset=FEATURE_SETS["B"] + ["target"])
    return out
