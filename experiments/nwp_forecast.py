"""Experiment: add archived weather-FORECAST features for the target hour (validation only).

Run from the repo root:  PYTHONPATH=src python experiments/nwp_forecast.py

Forecast file: archived_forecasts/site_a_and_b_lead1.csv (Open-Meteo "previous runs").
Lead 1 = the value predicted 24 h before the valid time. `valid_time_utc` is the END of the
hour it describes (same convention as the hourly weather rows), so the forecast for target
hour T (row label T, hour [T, T+1h)) sits in the forecast row with valid_time_utc = T + 1h.
It was issued about 24 h before that, i.e. at T - 23 h. Our issue time is T (end of hour t = T-1h),
so the forecast was already available ~23 h earlier: no leakage. Lead 0 is NOT used (it may be
issued after our issue time).
"""
import urllib.request
from pathlib import Path

import pandas as pd

from solarcast import config, data, evaluate, models
from solarcast.baselines import persistence
from solarcast.features import FEATURE_SETS, build_features

LEAD = 1
FC_WEATHER = ["temp_c", "humidity_pct", "cloud_cover_pct", "wind_speed_kmh", "ghi_wm2", "dni_wm2", "dhi_wm2"]
FC_COLS = [f"fc_{c}" for c in FC_WEATHER]
FC_EXTRA = ["fc_ghi_minus_ghi_now", "fc_cloud_minus_cloud_now"]  # forecast for target hour vs weather now


def load_forecast(lead: int = LEAD) -> pd.DataFrame:
    """Archived forecast indexed by the START of the hour it describes (UTC), columns fc_*."""
    name = f"site_a_and_b_lead{lead}.csv"
    path = Path(config.DATA_DIR) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(f"{config.DATASET_BASE_URL}/archived_forecasts/{name}", timeout=60).read())
    fc = pd.read_csv(path, parse_dates=["valid_time_utc"])
    fc.index = pd.DatetimeIndex(fc["valid_time_utc"]).tz_localize("UTC") - pd.Timedelta(hours=1)  # valid time = end of hour
    fc = fc[FC_WEATHER].rename(columns=lambda c: f"fc_{c}")
    return fc[~fc.index.duplicated()]


def add_forecast(feat: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Join the forecast for the target hour (index = target_time) and add 'forecast minus now' terms."""
    out = feat.join(fc, how="left")
    out["fc_ghi_minus_ghi_now"] = out["fc_ghi_wm2"] - out["ghi_wm2"]
    out["fc_cloud_minus_cloud_now"] = out["fc_cloud_cover_pct"] - out["cloud_cover_pct"]
    return out


def fit_predict(train, val, cols):
    """Fit the shared HistGradientBoosting model on `cols`; predict val (clipped to 0..capacity)."""
    m = models.make_model()
    m.fit(train[cols], train["target"])
    return pd.Series(m.predict(val[cols]).clip(0, config.CAPACITY_KW), index=val.index)


def run_site(site: str, fc: pd.DataFrame) -> pd.DataFrame:
    feat = add_forecast(build_features(data.load_site(site)), fc)
    sp = evaluate.split_by_time(feat)
    train, val = sp["train"], sp["val"]  # test split is never touched
    A, B = FEATURE_SETS["A"], FEATURE_SETS["B"]
    fc_all = FC_COLS + FC_EXTRA
    sets = {"Model A": A, "Model B": B, "Model C-lite (A+fc)": A + fc_all, "Model C (B+fc)": B + fc_all}
    preds = {"Persistence": persistence(val)}
    preds.update({name: fit_predict(train, val, cols) for name, cols in sets.items()})
    day = evaluate.daytime_mask(val)
    print(f"Site {site}: forecast NaN share in val daytime rows = {val.loc[day, FC_COLS].isna().any(axis=1).mean():.4f}")
    return evaluate.compare(val, preds)


if __name__ == "__main__":
    fc = load_forecast()
    # Sanity check on TRAIN only: forecast should track the measured GHI of the same hour.
    df = data.load_site("A")
    tr = df.loc["2025-01-01":"2025-12-31"]
    print("alignment check (train, corr of fc ghi vs measured ghi): "
          f"same hour = {tr['ghi_wm2'].corr(fc['fc_ghi_wm2'].reindex(tr.index)):.3f}, "
          f"shifted +1h = {tr['ghi_wm2'].corr(fc['fc_ghi_wm2'].reindex(tr.index + pd.Timedelta(hours=1)).set_axis(tr.index)):.3f}")
    for site in ("A", "B"):
        print(f"\n=== Site {site}, VALIDATION (Jan-Feb 2026), daytime ===")
        print(run_site(site, fc).round(3).to_string())
