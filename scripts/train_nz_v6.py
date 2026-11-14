"""SolarCast NZ v6 experiment: XGBoost trained on the GPU, a wider search, and a blend with v4.

scikit-learn's HistGradientBoosting (v4) only trains on the CPU. XGBoost trains the same kind of model (gradient-boosted
trees) on a CUDA GPU, so a much wider search fits in the same time. We keep v4's design: one model for raw next-hour
sunlight and one for the clear-sky index (cloudiness), averaged. Candidates:

  * XGBoost on the v4 inputs, and on v4 + the v5 inputs (day-ahead forecast error right now, forecast trend),
    over a grid of tree depth, learning rate, number of trees and regularisation;
  * the best XGBoost model blended with a v4 refit (different libraries make different errors).

Judged on the VALIDATION period only (Jan-Jun 2025), trained on 2022-2024 like v4. The bar is fixed in advance: more than
1 W/m2 better than the saved v4 model on validation (28.98 W/m2). The test period is not read here.

Usage (Colab with a T4):  PYTHONPATH=src python scripts/train_nz_v6.py   -> experiments/results/nz_v6.json
"""
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from solarcast.nz import config, data, model as nzmodel                         # noqa: E402
from solarcast.nz.ensemble import CS_MIN, K_MAX, SolarEnsemble                   # noqa: E402
from solarcast.nz.features import FEATURES, build_features                       # noqa: E402
from train_nz_v5 import ERR_FEATURES, TREND_FEATURES, add_extra, extra_features, score, split   # noqa: E402

MIN_GAIN_WM2 = 1.0


def frame(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = feat[cols].copy()
    x["region_id"] = pd.Categorical(x["region_id"].astype(int), categories=range(len(config.REGIONS)))
    return x


def xgb_model(params: dict, objective: str) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(tree_method="hist", device="cuda", enable_categorical=True, max_cat_to_onehot=1,
                            objective=objective, random_state=0, n_jobs=4, **params)


def fit_pair(train: pd.DataFrame, cols: list[str], params: dict):
    """v4's two models with XGBoost: raw GHI (absolute error, sun up) and clear-sky index (squared error)."""
    known = train["target_ghi"].notna()
    up = train[(train["elev_target"] > 0) & known]
    raw = xgb_model(params, "reg:absoluteerror").fit(frame(up, cols), up["target_ghi"])
    hi = train[(train["cs_target"] >= CS_MIN) & known]
    kidx = xgb_model(params, "reg:squarederror").fit(frame(hi, cols), (hi["target_ghi"] / hi["cs_target"]).clip(0, K_MAX))
    return raw, kidx


def predict_pair(raw, kidx, val: pd.DataFrame, cols: list[str]):
    x = frame(val, cols)
    up = val["elev_target"].to_numpy() > 0
    r = np.where(up, np.clip(raw.predict(x), 0, None), 0.0)
    cs = val["cs_target"].to_numpy()
    k = np.where(up, np.where(cs >= CS_MIN, np.clip(kidx.predict(x), 0, K_MAX) * cs, r), 0.0)
    return r, k


def main():
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() or "NO GPU", flush=True)
    meta = nzmodel.load_meta()
    v4_val = meta["metrics"]["val"]["overall"]["model"]["mae_wm2"]
    long_df = data.fetch_history(config.HISTORY_START, meta.get("data_end", "2026-11-08"))
    feat = add_extra(build_features(long_df), extra_features(long_df))
    train, val = split(feat, "train"), split(feat, "val")
    print(f"train rows {len(train)}, val rows {len(val)}", flush=True)

    t0 = time.time()
    v4 = SolarEnsemble(FEATURES, nzmodel.V1_PARAMS).fit(train)
    v4_pred = v4.predict(val)
    results = {"v4 refit (HistGradientBoosting, CPU)": {**score(val, v4_pred), "seconds": round(time.time() - t0)}}
    print(f"VAL v4 refit {results['v4 refit (HistGradientBoosting, CPU)']}", flush=True)

    feature_sets = {"v4 inputs": FEATURES, "v4 + v5 inputs": FEATURES + ERR_FEATURES + TREND_FEATURES}
    grid = [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=mcw, subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=1.0)
            for d, lr, n, mcw in itertools.product((6, 8, 10), (0.03, 0.06), (800, 2000), (20, 100))]
    preds = {}
    for (fs_name, cols), params in itertools.product(feature_sets.items(), grid):
        name = (f"XGBoost GPU, {fs_name}, depth {params['max_depth']}, lr {params['learning_rate']}, "
                f"{params['n_estimators']} trees, min_child_weight {params['min_child_weight']}")
        t0 = time.time()
        raw, kidx = fit_pair(train, cols, params)
        r, k = predict_pair(raw, kidx, val, cols)
        preds[name] = (r + k) / 2
        results[name] = {**score(val, preds[name]), "seconds": round(time.time() - t0)}
        print(f"VAL {results[name]['val_mae_wm2']:.2f} W/m2 ({results[name]['seconds']} s)  {name}", flush=True)

    best_xgb = min(preds, key=lambda k: results[k]["val_mae_wm2"])
    blends = {f"{w:.1f}": score(val, w * preds[best_xgb] + (1 - w) * v4_pred)["val_mae_wm2"]
              for w in np.arange(0.0, 1.01, 0.1)}
    w_best = min(blends, key=blends.get)
    print(f"blend weight of XGBoost (rest v4 refit): {blends}", flush=True)
    results[f"blend: {w_best} x best XGBoost + rest v4 refit"] = {"val_mae_wm2": blends[w_best]}
    best = min(results, key=lambda k: results[k]["val_mae_wm2"])
    best_mae = results[best]["val_mae_wm2"]
    gain = v4_val - best_mae
    verdict = "PROMOTE" if gain > MIN_GAIN_WM2 else "NOT PROMOTED: gain on validation below the 1 W/m2 bar; keeping v4"
    print(f"\nbest: {best}: {best_mae:.2f} vs saved v4 {v4_val:.2f} W/m2 -> gain {gain:.2f}\n{verdict}", flush=True)
    out = {"v4_val_mae_wm2": round(v4_val, 3), "bar_wm2": MIN_GAIN_WM2, "gpu": True, "candidates": results,
           "best_xgboost": best_xgb, "blend_weight_search": blends, "best": best, "best_val_mae_wm2": best_mae,
           "gain_wm2": round(gain, 3), "verdict": verdict, "test_period_read": False,
           "xgboost_version": xgb.__version__}
    (ROOT / "experiments" / "results" / "nz_v6.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
