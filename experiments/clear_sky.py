"""Experiment: clear-sky normalisation of the target. Train on TRAIN only, report VAL only.

Run:  PYTHONPATH=src python experiments/clear_sky.py
Two clear-sky estimates, both learned from the TRAIN split only:
  "env": per (day-of-year +-WINDOW days, local hour) 95th percentile of train power
  "sin": Q * sin(solar elevation), Q = 95th percentile of power / sin(elevation) in train
"""
import numpy as np
import pandas as pd

from solarcast import config, data, evaluate, models
from solarcast.baselines import persistence
from solarcast.features import FEATURE_SETS, build_features

WINDOW, QUANTILE = 10, 0.95       # envelope: +-10 days around the day of year, 95th percentile
MIN_CSP = 2.0                     # kW. Below this the clear-sky estimate is too small to divide by
K_MAX = 1.5                       # clip the clear-sky index k = power / clear_sky_power
CS_FEATURES = ["csp_target", "csp_issue", "ci_now", "ci_roll3"]


def _doy_hour(times):
    """Day of year (1..366) and local hour of each timestamp."""
    local = times.tz_convert(config.LOCAL_TZ)
    return local.day_of_year.to_numpy(), local.hour.to_numpy()


def fit_envelope(train):
    """Table env[doy, hour] = high quantile of train power (kW). NaN where train has no data."""
    doy, hour = _doy_hour(train.index)
    power = train["target"].to_numpy()
    env = np.full((367, 24), np.nan)
    for h in range(24):
        d, p = doy[hour == h], power[hour == h]
        for day in range(1, 367):
            gap = np.abs(d - day)
            near = np.minimum(gap, 366 - gap) <= WINDOW      # circular distance in days
            if near.sum() >= 5:
                env[day, h] = np.quantile(p[near], QUANTILE)
    return lambda times: env[_doy_hour(times)]               # times -> clear-sky kW


def fit_sin(train):
    """csp = Q * sin(elevation). Q learned from train daytime rows."""
    day = train[evaluate.daytime_mask(train)]
    s = np.sin(np.radians(day["target_solar_elevation"]))
    q = np.quantile(day["target"] / s, QUANTILE)
    return lambda times, elev: q * np.sin(np.radians(np.clip(elev, 0, None)))


def add_cs(feat, kind, csp_train_fn):
    """Add clear-sky columns. Issue hour = target time - 1 h (only calendar/sun, no leakage)."""
    out = feat.copy()
    if kind == "env":
        out["csp_target"] = csp_train_fn(feat.index)
        out["csp_issue"] = csp_train_fn(feat.index - pd.Timedelta(hours=1))
    else:
        out["csp_target"] = csp_train_fn(feat.index, feat["target_solar_elevation"].to_numpy())
        out["csp_issue"] = csp_train_fn(feat.index, feat["issue_solar_elevation"].to_numpy())
    safe = out["csp_issue"].where(out["csp_issue"] >= MIN_CSP)   # NaN when sun is too low
    out["ci_now"] = (out["power_lag1"] / safe).clip(0, K_MAX)
    out["ci_roll3"] = (out["power_roll3_mean"] / safe).clip(0, K_MAX)
    return out


def fit_kw(train, cols):
    """Ordinary model: predict kW directly."""
    m = models.make_model().fit(train[cols], train["target"])
    return lambda f: pd.Series(m.predict(f[cols]).clip(0, config.CAPACITY_KW), index=f.index)


def fit_k(train, cols, fallback):
    """Predict k = kW / clear_sky_kW, convert back; use `fallback` where clear sky is tiny."""
    tr = train[train["csp_target"] >= MIN_CSP]
    k = (tr["target"] / tr["csp_target"]).clip(0, K_MAX)
    # weight by csp: an error of dk on k is dk * csp kW, so this matches the kW MAE
    m = models.make_model().fit(tr[cols], k, sample_weight=tr["csp_target"])

    def predict(f):
        pred = pd.Series(m.predict(f[cols]) * f["csp_target"], index=f.index)
        pred = pred.where(f["csp_target"] >= MIN_CSP, fallback(f))
        return pred.clip(0, config.CAPACITY_KW)
    return predict


def run_site(site):
    feat = build_features(data.load_site(site))
    parts = evaluate.split_by_time(feat)
    train, val = parts["train"], parts["val"]               # the test split is never touched
    a, b = FEATURE_SETS["A"], FEATURE_SETS["B"]

    env_fn, sin_fn = fit_envelope(train), fit_sin(train)
    tr_e, va_e = add_cs(train, "env", env_fn), add_cs(val, "env", env_fn)
    tr_s, va_s = add_cs(train, "sin", sin_fn), add_cs(val, "sin", sin_fn)

    model_b = fit_kw(train, b)                               # raw-kW baseline = fallback
    preds = {
        "Persistence": persistence(val),
        "Model A": fit_kw(train, a)(val),
        "Model B": model_b(val),
        "A + CS feats (env)": fit_kw(tr_e, a + CS_FEATURES)(va_e),
        "B + CS feats (env)": fit_kw(tr_e, b + CS_FEATURES)(va_e),
        "B + CS feats (sin)": fit_kw(tr_s, b + CS_FEATURES)(va_s),
        "B k-index (env)": fit_k(tr_e, b, model_b)(va_e),
        "B + CS feats k-index (env)": fit_k(tr_e, b + CS_FEATURES, model_b)(va_e),
        "B k-index (sin)": fit_k(tr_s, b, model_b)(va_s),
    }
    table = evaluate.compare(val, preds).round(3)
    print(f"\n=== Site {site}, VAL daytime (Jan-Feb 2026), n rows = {int(table['n'].iloc[0])} ===")
    print(table.to_string())
    best = table.drop(index=["Persistence", "Model A", "Model B"]).index[0]
    print(f"\nBest variant: {best}. Site {site} val MAE by sky: {best} vs Model B")
    g = evaluate.grouped_metrics(val, {"Model B": preds["Model B"], best: preds[best]}, by="sky")
    print(g.round(3).to_string(index=False))
    return table


if __name__ == "__main__":
    for s in ("A", "B"):
        run_site(s)
