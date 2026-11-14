"""SolarCast NZ v4 experiment: predict the CLEAR-SKY INDEX instead of raw sunlight, and average models.

Most of the hour-to-hour change in sunlight is the sun moving, which astronomy already knows exactly. If the model
predicts the clear-sky index k = GHI / clear-sky GHI (how cloudy it is) and we multiply back by the target hour's
clear-sky GHI, it only has to learn the clouds. We also test averaging the raw-GHI model and the k model (a small
ensemble). Inputs: v1 features plus the day-ahead forecast from v3 (train 2022-2024; forecasts exist from 2024).

Judged on the VALIDATION period only, against the saved v1 model; a gain above 1 W/m2 is required to promote.
Usage (run on Colab, after v3 cached the data):  PYTHONPATH=src python scripts/train_nz_v4.py
(Experiment record: v4 was PROMOTED; scripts/train_nz.py now trains and saves it with solarcast.nz.ensemble.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from solarcast.nz import config, data, model as nzmodel                 # noqa: E402
from solarcast.nz.features import FORECAST_FEATURES as FC_FEATURES, V1_FEATURES as FEATURES, build_features  # noqa: E402
from train_nz_v3 import V1_PARAMS, fit, predict, split  # noqa: E402

CS_MIN = 20.0   # W/m2: below this (dawn/dusk) the index is unstable, so the raw-GHI model is used
MIN_GAIN_WM2 = 1.0


def fit_k(feat, cols, params):
    """Model of the clear-sky index of the target hour."""
    rows = feat[(feat["cs_target"] >= CS_MIN) & feat["target_ghi"].notna()]
    k = (rows["target_ghi"] / rows["cs_target"]).clip(0, 1.5)
    m = HistGradientBoostingRegressor(categorical_features=[cols.index("region_id")], random_state=0,
                                      **{**nzmodel.DEFAULT_PARAMS, **params, "loss": "squared_error"})
    return m.fit(rows[cols], k)


def predict_k(m_k, m_raw, feat, cols_k, cols_raw):
    """k * clear-sky GHI where the sun is high enough, otherwise the raw model's GHI."""
    raw = predict(m_raw, feat, cols_raw)
    kghi = np.clip(m_k.predict(feat[cols_k]), 0, 1.5) * feat["cs_target"]
    out = np.where(feat["cs_target"] >= CS_MIN, kghi, raw)
    return pd.Series(np.where(feat["elev_target"] > 0, out, 0.0), index=feat.index)


def main():
    v1_meta = nzmodel.load_meta()
    end = v1_meta.get("data_end", "2026-11-08")
    feat = build_features(data.fetch_history(config.HISTORY_START, end))      # includes the fc_* forecast features
    train, val = split(feat, "train"), split(feat, "val")
    cols_fc = FEATURES + FC_FEATURES
    base = {"persistence": nzmodel.persistence(val), "clear_sky": nzmodel.clear_sky_persistence(val)}

    raw_v1 = fit(train, FEATURES, V1_PARAMS)
    raw_fc = fit(train, cols_fc, V1_PARAMS)
    k_v1 = fit_k(train, FEATURES, V1_PARAMS)
    k_fc = fit_k(train, cols_fc, V1_PARAMS)
    preds = {
        "raw GHI, v1 features (= v1)": predict(raw_v1, val, FEATURES),
        "raw GHI, + forecast (= v3 best)": predict(raw_fc, val, cols_fc),
        "clear-sky index, v1 features": predict_k(k_v1, raw_v1, val, FEATURES, FEATURES),
        "clear-sky index, + forecast": predict_k(k_fc, raw_fc, val, cols_fc, cols_fc),
    }
    preds["average of raw + k, + forecast"] = (preds["raw GHI, + forecast (= v3 best)"] + preds["clear-sky index, + forecast"]) / 2
    results = {}
    for name, p in preds.items():
        ev = nzmodel.evaluate(val, {"model": p, **base})
        results[name] = {"val_mae_wm2": round(ev["overall"]["model"]["mae_wm2"], 3),
                         "val_mae_kw": round(ev["overall"]["model"]["mae_kw"], 4)}
        print(f"VAL {name:34s} {results[name]['val_mae_wm2']:.2f} W/m2  {results[name]['val_mae_kw']:.3f} kW", flush=True)
    v1_val = v1_meta["metrics"]["val"]["overall"]["model"]["mae_wm2"]
    best = min(results, key=lambda k: results[k]["val_mae_wm2"])
    gain = v1_val - results[best]["val_mae_wm2"]
    verdict = "PROMOTE" if gain > MIN_GAIN_WM2 else "NOT PROMOTED: gain on validation too small; keeping v1"
    print(f"\nbest: {best} ({results[best]['val_mae_wm2']:.2f}) vs v1 saved {v1_val:.2f} W/m2 -> gain {gain:.2f}\n{verdict}")
    Path("experiments/results/nz_v4.json").write_text(json.dumps(
        {"v1_val_mae_wm2": v1_val, "candidates": results, "best": best, "gain_wm2": round(gain, 3),
         "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
