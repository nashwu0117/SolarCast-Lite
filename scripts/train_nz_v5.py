"""SolarCast NZ v5 experiment: learn from how wrong the day-ahead forecast is RIGHT NOW, then tune.

Idea. v4 already reads Open-Meteo's day-ahead forecast for the target hour (issued about 24 h earlier). At issue time t we
also know the day-ahead forecast that was made for hour t itself, and we know what really happened in hour t. Their
difference tells the model whether today's forecast is running too sunny or too cloudy, so it can correct the target-hour
forecast. Every new input below was issued about a day before t or is observed at or before t (no leakage):

  fc_ghi_now     day-ahead forecast GHI for stamp t         fc_err_now     ghi_t - fc_ghi_now
  fc_kt_now      fc_ghi_now / clear-sky GHI at t            kt_err_now     kt_t - fc_kt_now
  fc_err_roll3   mean of fc_err over the last 3 hours       cloud_err_now  cloud_t - day-ahead cloud for t
  fc_ghi_next    day-ahead forecast GHI for stamp t+2       fc_d_ghi       fc_ghi_target - fc_ghi_now
  (fc_ghi_next and fc_d_ghi: the forecast's own trend around the target hour; also issued ~24 h before.)

Candidates are judged on the VALIDATION period only (Jan-Jun 2025), trained on 2022-2024 like v4. The bar is fixed in
advance: more than 1 W/m2 better than the saved v4 model on validation (28.98 W/m2). The test period is not read here.

Usage (run on Colab):  PYTHONPATH=src python scripts/train_nz_v5.py   -> experiments/results/nz_v5.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarcast.nz import config, data, model as nzmodel                         # noqa: E402
from solarcast.nz.ensemble import SolarEnsemble                                  # noqa: E402
from solarcast.nz.features import CS_MIN, FEATURES, HOUR, build_features         # noqa: E402

MIN_GAIN_WM2 = 1.0
ERR_FEATURES = ["fc_ghi_now", "fc_err_now", "fc_kt_now", "kt_err_now", "fc_err_roll3", "cloud_err_now"]
TREND_FEATURES = ["fc_ghi_next", "fc_d_ghi"]


def extra_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """The v5 inputs, indexed like build_features: (region, target_time = t + 1h)."""
    parts = []
    for region, g in long_df.groupby("region", sort=False):
        g = g.sort_values("time_utc").set_index("time_utc")
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC"))
        f = pd.DataFrame(index=g.index)
        fc = g["fc_ghi_wm2"]
        f["fc_ghi_now"] = fc
        f["fc_err_now"] = g["ghi_wm2"] - fc
        f["fc_err_roll3"] = f["fc_err_now"].rolling(3, min_periods=2).mean()
        f["cloud_err_now"] = g["cloud_cover_pct"] - g["fc_cloud_pct"]
        f["fc_ghi_next"] = fc.shift(-2)
        f["region"], f["target_time"] = region, f.index + HOUR
        parts.append(f)
    return pd.concat(parts).set_index(["region", "target_time"]).sort_index()


def add_extra(feat: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    out = feat.join(extra, how="left")
    out["fc_kt_now"] = (out["fc_ghi_now"] / out["cs_t"]).where(out["cs_t"] >= CS_MIN).clip(0, 1.5)
    out["kt_err_now"] = out["kt_t"] - out["fc_kt_now"]
    out["fc_d_ghi"] = out["fc_ghi_target"] - out["fc_ghi_now"]
    return out


def split(feat: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS[name])
    tm = feat.index.get_level_values("target_time")
    return feat[(tm >= a) & (tm < b)]


def score(val: pd.DataFrame, pred: np.ndarray) -> dict:
    ev = nzmodel.evaluate(val, {"model": pd.Series(pred, index=val.index)})
    return {"val_mae_wm2": round(ev["overall"]["model"]["mae_wm2"], 3), "val_mae_kw": round(ev["overall"]["model"]["mae_kw"], 4)}


def main():
    meta = nzmodel.load_meta()
    v4_val = meta["metrics"]["val"]["overall"]["model"]["mae_wm2"]
    long_df = data.fetch_history(config.HISTORY_START, meta.get("data_end", "2026-11-08"))
    feat = add_extra(build_features(long_df), extra_features(long_df))
    train, val = split(feat, "train"), split(feat, "val")
    print(f"train rows {len(train)}, val rows {len(val)}", flush=True)

    v1 = nzmodel.V1_PARAMS
    f_err, f_all = FEATURES + ERR_FEATURES, FEATURES + ERR_FEATURES + TREND_FEATURES
    cands = {
        "v4 refit (sanity check)": (FEATURES, v1),
        "v4 + forecast-error-now": (f_err, v1),
        "v4 + forecast-error-now + forecast trend": (f_all, v1),
        "all new inputs, faster learning (lr 0.05, 1000 trees)": (f_all, dict(v1, learning_rate=0.05, max_iter=1000)),
        "all new inputs, bigger trees (127 leaves, 1500 trees)": (f_all, dict(v1, max_iter=1500, max_leaf_nodes=127, min_samples_leaf=200)),
        "all new inputs, more regularised (l2 3, leaf 200)": (f_all, dict(v1, l2_regularization=3.0, min_samples_leaf=200)),
    }
    results, models = {}, {}
    for name, (cols, params) in cands.items():
        t0 = time.time()
        m = SolarEnsemble(cols, params).fit(train)
        raw, kpart = m.predict_parts(val)
        results[name] = {**score(val, (raw + kpart) / 2), "features": len(cols), "params": params,
                         "seconds": round(time.time() - t0)}
        models[name] = (raw, kpart)
        print(f"VAL {name:58s} {results[name]['val_mae_wm2']:.2f} W/m2  {results[name]['val_mae_kw']:.3f} kW "
              f"({results[name]['seconds']} s)", flush=True)

    best = min((k for k in results if not k.startswith("v4 refit")), key=lambda k: results[k]["val_mae_wm2"])
    raw, kpart = models[best]
    weights = {f"{w:.1f}": score(val, w * raw + (1 - w) * kpart)["val_mae_wm2"] for w in np.arange(0.2, 0.81, 0.1)}
    print(f"weight of the raw model for '{best}': {weights}", flush=True)
    w_best = min(weights, key=weights.get)
    best_mae = min(results[best]["val_mae_wm2"], weights[w_best])
    gain = v4_val - best_mae
    verdict = "PROMOTE" if gain > MIN_GAIN_WM2 else "NOT PROMOTED: gain on validation below the 1 W/m2 bar; keeping v4"
    print(f"\nbest: {best} (raw weight {w_best}) {best_mae:.2f} vs saved v4 {v4_val:.2f} W/m2 -> gain {gain:.2f}\n{verdict}")
    out = {"v4_val_mae_wm2": round(v4_val, 3), "bar_wm2": MIN_GAIN_WM2, "candidates": results, "best": best,
           "raw_weight_search": weights, "best_raw_weight": float(w_best), "best_val_mae_wm2": best_mae,
           "gain_wm2": round(gain, 3), "verdict": verdict, "test_period_read": False}
    (ROOT / "experiments" / "results" / "nz_v5.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
