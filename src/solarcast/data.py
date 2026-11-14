"""Download and load the hourly solar + weather CSVs into one clean, complete hourly table."""
import urllib.request
from pathlib import Path

import pandas as pd

from solarcast import config


def download_site_csv(site: str, force: bool = False) -> Path:
    """Download a site's CSV into config.DATA_DIR (skipped if already cached)."""
    rel_path = config.SITES[site]["csv"]
    dest = Path(config.DATA_DIR) / Path(rel_path).name
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{config.DATASET_BASE_URL}/{rel_path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        dest.write_bytes(response.read())
    return dest


def load_site(site: str) -> pd.DataFrame:
    """Load a site as a clean table: UTC hourly index, sorted, unique, no missing hours.

    Missing measurements stay NaN (never filled with 0); night zeros are real zeros.
    """
    raw = pd.read_csv(download_site_csv(site))
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw["datetime_utc"]), name="datetime_utc").tz_localize("UTC")
    raw = raw.sort_index()

    n_duplicates = int(raw.index.duplicated().sum())
    raw = raw[~raw.index.duplicated(keep="first")]

    full_grid = pd.date_range(raw.index[0], raw.index[-1], freq="h", tz="UTC", name="datetime_utc")
    df = raw[config.KEEP_COLS].astype(float).reindex(full_grid)  # missing hours become NaN rows

    df.attrs["n_duplicate_timestamps_in_csv"] = n_duplicates
    df.attrs["n_inserted_gap_rows"] = len(full_grid) - len(raw)
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summarise size, date range, missing values and daytime share of a load_site table."""
    daytime = df["solar_elevation_deg"] > config.DAYTIME_ELEVATION_MIN
    return {
        "n_rows": len(df),
        "start": df.index[0],
        "end": df.index[-1],
        "n_missing_power": int(df[config.TARGET_COL].isna().sum()),
        "n_missing_weather": int(df[config.WEATHER_COLS].isna().any(axis=1).sum()),
        "n_duplicate_timestamps_in_csv": df.attrs.get("n_duplicate_timestamps_in_csv", 0),
        "n_inserted_gap_rows": df.attrs.get("n_inserted_gap_rows", 0),
        "pct_daytime": round(100 * float(daytime.mean()), 1),
    }
