"""[ARCHIVED experiment, written against the v1 model code (one gradient-boosting model, 29 features); the results are
in the matching experiments/results file. nzmodel.train now returns the v4 ensemble, so a rerun needs the v1 code.]
SolarCast NZ v2 experiment: does training on MORE New Zealand locations make a stronger model?

v1 trained on the 14 dashboard regions only, and its learning curve flattened after about a year, so more years of
the same sites add little. Here we add 16 extra sites spread over both islands as TRAINING data. Everything is
judged on the original 14 regions, so the numbers stay comparable with v1 and the dashboard is unchanged:
the extra sites get region ids 14-29, while the 14 dashboard regions keep ids 0-13.

Protocol: compare on the VALIDATION period only. Only if the best candidate beats v1 on validation by more than
1 W/m2 do we look at the TEST period, once, and save it as the new model (v1's metrics are kept under "previous").

Usage (run on Colab):  PYTHONPATH=src python scripts/train_nz_v2.py [--end YYYY-MM-DD]
"""
import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarcast.nz import config, data, model as nzmodel          # noqa: E402
from solarcast.nz.features import build_features                 # noqa: E402

EXTRA_SITES = {  # training-only locations (lat, lon)
    "Kaitaia": (-35.11, 173.26), "Whakatane": (-37.95, 176.99), "Rotorua": (-38.14, 176.25),
    "Taupo": (-38.69, 176.07), "Masterton": (-40.95, 175.66), "Blenheim": (-41.51, 173.95),
    "Westport": (-41.75, 171.60), "Greymouth": (-42.45, 171.21), "Hokitika": (-42.72, 170.97),
    "Kaikoura": (-42.40, 173.68), "Timaru": (-44.40, 171.25), "Oamaru": (-45.10, 170.97),
    "Twizel": (-44.26, 170.10), "Te Anau": (-45.41, 167.72), "Alexandra": (-45.25, 169.39),
    "Ashburton": (-43.90, 171.75),
}
DASHBOARD = list(config.REGIONS)            # the original 14, in their original order (ids 0-13)
V1_PARAMS = dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=63, min_samples_leaf=100, loss="absolute_error")
CANDIDATES = {
    "v1 config, 14 sites": (False, V1_PARAMS),
    "v1 config, 30 sites": (True, V1_PARAMS),
    "deeper, 30 sites": (True, dict(V1_PARAMS, max_iter=1200, max_leaf_nodes=127, min_samples_leaf=200)),
    "faster lr, 30 sites": (True, dict(V1_PARAMS, learning_rate=0.06, max_iter=800, max_leaf_nodes=63)),
}
MIN_GAIN_WM2 = 1.0


def split(feat, name, regions=None):
    a, b = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS[name])
    tm = feat.index.get_level_values("target_time")
    out = feat[(tm >= a) & (tm < b)]
    if regions is not None:
        out = out[out.index.get_level_values("region").isin(regions)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=nzmodel.load_meta().get("data_end") or (date.today() - timedelta(days=6)).isoformat(),
                    help="defaults to v1's data_end so the comparison uses the same data")
    end = ap.parse_args().end
    val_end = (pd.Timestamp(config.SPLITS["val"][1]) - pd.Timedelta(days=1)).date().isoformat()

    config.REGIONS = {**config.REGIONS, **EXTRA_SITES}   # extra sites appended -> ids 14-29
    base = data.fetch_history(config.HISTORY_START, end, regions=DASHBOARD)
    extra = data.fetch_history(config.HISTORY_START, val_end, regions=list(EXTRA_SITES))
    feat = build_features(pd.concat([base, extra], ignore_index=True))
    train_all = split(feat, "train")
    train_14 = split(feat, "train", DASHBOARD)
    val, test = split(feat, "val", DASHBOARD), split(feat, "test", DASHBOARD)
    print("rows train(14)/train(30)/val/test:", len(train_14), len(train_all), len(val), len(test), flush=True)

    baselines = {"persistence": nzmodel.persistence(val), "clear_sky": nzmodel.clear_sky_persistence(val)}
    results = {}
    for name, (more_sites, params) in CANDIDATES.items():
        m = nzmodel.train(train_all if more_sites else train_14, **params)
        ev = nzmodel.evaluate(val, {"model": nzmodel.predict(m, val), **baselines})
        results[name] = (ev["overall"]["model"]["mae_wm2"], ev["overall"]["model"]["mae_kw"], m, ev, params)
        print(f"VAL {name:24s} MAE {results[name][0]:.2f} W/m2  {results[name][1]:.3f} kW", flush=True)
    o = next(iter(results.values()))[3]["overall"]
    print(f"VAL baselines: persistence {o['persistence']['mae_wm2']:.2f}, clear-sky {o['clear_sky']['mae_wm2']:.2f} W/m2")

    v1_meta = nzmodel.load_meta()
    v1_val = v1_meta["metrics"]["val"]["overall"]["model"]["mae_wm2"]
    best_name = min(results, key=lambda k: results[k][0])
    best_mae, _, model, val_ev, params = results[best_name]
    print(f"\nbest: {best_name} {best_mae:.2f} W/m2 vs v1 saved {v1_val:.2f} W/m2", flush=True)
    summary = {k: {"val_mae_wm2": round(v[0], 3), "val_mae_kw": round(v[1], 4)} for k, v in results.items()}
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/nz_v2.json").write_text(json.dumps({"v1_val_mae_wm2": v1_val, "candidates": summary,
                                                                  "best": best_name}, indent=2))
    if v1_val - best_mae <= MIN_GAIN_WM2:
        print("NOT PROMOTED: gain on validation is too small; keeping v1.")
        return

    test_ev = nzmodel.evaluate(test, nzmodel.all_predictions(model, test))   # the ONE look at test
    t = test_ev["overall"]
    print(f"TEST (once): model {t['model']['mae_wm2']:.2f} W/m2 {t['model']['mae_kw']:.3f} kW; "
          f"persistence {t['persistence']['mae_wm2']:.2f}; clear-sky {t['clear_sky']['mae_wm2']:.2f}", flush=True)
    config.REGIONS = {k: config.REGIONS[k] for k in DASHBOARD}   # report on the dashboard regions only
    report = nzmodel.build_report(model, val, test)
    previous = {"version": v1_meta.get("version", "v1"), "metrics": v1_meta["metrics"],
                "chosen_params": v1_meta.get("chosen_params")}
    nzmodel.save(model, {"val": val_ev, "test": test_ev},
                 extra={**{k: v1_meta[k] for k in ("learning_curve",) if k in v1_meta},
                        "version": "v2", "chosen_params": params, "data_end": end, "report": report,
                        "changes": f"{best_name}: +16 training-only NZ sites ({', '.join(EXTRA_SITES)})",
                        "training_sites": DASHBOARD + list(EXTRA_SITES), "candidates": summary,
                        "previous": previous, "data_source": "Open-Meteo (ERA5 reanalysis), CC BY 4.0"})
    print("PROMOTED to v2; saved", config.MODEL_PATH)


if __name__ == "__main__":
    main()
