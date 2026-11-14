"""SolarCast NZ v3 experiment: does a day-ahead WEATHER FORECAST for the target hour make the model stronger?

v1 only knows the weather up to now. v2 showed that more rows of the same kind no longer help, so v3 adds new
information: Open-Meteo's archived forecast for the target hour, from the Previous Runs API (`*_previous_day1`
variables = the forecast issued about 24 hours earlier). It existed well before our issue time t, so there is no
leakage, and a live system can get the same numbers from the forecast API.

Archived day-ahead forecasts only exist from 2024, so the fair comparison trains every candidate on 2024 (the
learning curve showed one year of history is already close to the full-history result). Judged on the VALIDATION
period only; only a gain of more than 1 W/m2 over the saved v1 model would justify one look at the test period.

Usage (run on Colab):  PYTHONPATH=src python scripts/train_nz_v3.py
(Experiment record: the forecast download and the fc_* features now live in solarcast.nz.data / features, because v4
promoted them; the numbers in experiments/results/nz_v3.json were produced with the identical logic.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarcast.nz import config, data, model as nzmodel          # noqa: E402
from solarcast.nz.features import FORECAST_FEATURES as FC_FEATURES, V1_FEATURES as FEATURES, build_features  # noqa: E402

FC_START = config.FORECAST_START
V1_PARAMS = dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=63, min_samples_leaf=100, loss="absolute_error")
MIN_GAIN_WM2 = 1.0


def fit(feat, cols, params):
    rows = feat[(feat["elev_target"] > 0) & feat["target_ghi"].notna()]
    m = HistGradientBoostingRegressor(categorical_features=[cols.index("region_id")], random_state=0,
                                      **{**nzmodel.DEFAULT_PARAMS, **params})
    return m.fit(rows[cols], rows["target_ghi"])


def predict(m, feat, cols):
    ghi = np.clip(m.predict(feat[cols]), 0, None)
    return pd.Series(np.where(feat["elev_target"] > 0, ghi, 0.0), index=feat.index)


def split(feat, name, start=None):
    a, b = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS[name])
    a = max(a, pd.Timestamp(start, tz="UTC")) if start else a
    tm = feat.index.get_level_values("target_time")
    return feat[(tm >= a) & (tm < b)]


def main():
    v1_meta = nzmodel.load_meta()
    end = v1_meta.get("data_end", "2026-11-08")
    feat = build_features(data.fetch_history(config.HISTORY_START, end))      # includes the fc_* forecast features
    val = split(feat, "val")
    print("forecast coverage on val:", round(val["fc_ghi_target"].notna().mean() * 100, 1), "%", flush=True)
    print("corr(fc_ghi_target, target_ghi) on val daytime:",
          round(val.loc[val["elev_target"] > 0, ["fc_ghi_target", "target_ghi"]].corr().iloc[0, 1], 3), flush=True)

    base = {"persistence": nzmodel.persistence(val), "clear_sky": nzmodel.clear_sky_persistence(val)}
    cands = {
        "v1 features, train 2024": (FEATURES, split(feat, "train", FC_START), V1_PARAMS),
        "v1 + forecast, train 2024": (FEATURES + FC_FEATURES, split(feat, "train", FC_START), V1_PARAMS),
        "v1 + forecast, train 2022-2024": (FEATURES + FC_FEATURES, split(feat, "train"), V1_PARAMS),
        "v1 + forecast, train 2024, deeper": (FEATURES + FC_FEATURES, split(feat, "train", FC_START),
                                              dict(V1_PARAMS, max_iter=1200, max_leaf_nodes=127, min_samples_leaf=200)),
    }
    results = {}
    for name, (cols, train, params) in cands.items():
        m = fit(train, cols, params)
        ev = nzmodel.evaluate(val, {"model": predict(m, val, cols), **base})
        results[name] = {"val_mae_wm2": round(ev["overall"]["model"]["mae_wm2"], 3),
                         "val_mae_kw": round(ev["overall"]["model"]["mae_kw"], 4), "train_rows": len(train)}
        print(f"VAL {name:36s} {results[name]['val_mae_wm2']:.2f} W/m2  {results[name]['val_mae_kw']:.3f} kW", flush=True)
    o = ev["overall"]
    print(f"VAL baselines: persistence {o['persistence']['mae_wm2']:.2f}, clear-sky {o['clear_sky']['mae_wm2']:.2f} W/m2")
    v1_val = v1_meta["metrics"]["val"]["overall"]["model"]["mae_wm2"]
    best = min(results, key=lambda k: results[k]["val_mae_wm2"])
    gain = v1_val - results[best]["val_mae_wm2"]
    verdict = "PROMOTE (gain > 1 W/m2; integrate forecasts into service before saving)" if gain > MIN_GAIN_WM2 \
        else "NOT PROMOTED: gain on validation too small; keeping v1"
    print(f"\nbest: {best} ({results[best]['val_mae_wm2']:.2f}) vs v1 saved {v1_val:.2f} W/m2 -> gain {gain:.2f}\n{verdict}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/nz_v3.json").write_text(json.dumps(
        {"v1_val_mae_wm2": v1_val, "candidates": results, "best": best, "gain_wm2": round(gain, 3), "verdict": verdict,
         "forecast_source": "Open-Meteo Previous Runs API, *_previous_day1 (issued ~24 h earlier), from 2024"}, indent=2))


if __name__ == "__main__":
    main()
