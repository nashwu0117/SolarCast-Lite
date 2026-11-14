"""Train SolarCast NZ (v4): fetch history + day-ahead forecasts, build features, train on TRAIN, check VAL, report TEST once.

v4 = the average of a raw-GHI model and a clear-sky-index model (solarcast.nz.ensemble), both fed with the v1 features plus
Open-Meteo's day-ahead forecast for the target hour. It was chosen in scripts/train_nz_v2.py .. train_nz_v4.py by a bar fixed
in advance: more than 1 W/m2 better than v1 on the validation period (Jan-Jun 2025). Settings are v1's tuned ones (no search here).

Usage (run on Colab):  PYTHONPATH=src python scripts/train_nz.py [--end YYYY-MM-DD] [--force]
Writes models/nz_model.joblib and models/nz_model.json (metrics, dashboard report, experiment chain, v1 as "previous").
"""
import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarcast.nz import config, data, model as nzmodel          # noqa: E402
from solarcast.nz.features import build_features                 # noqa: E402

MIN_GAIN_WM2 = 1.0           # the promotion bar, fixed in advance
VERSION = "v4"
CHANGES = ("Day-ahead weather forecast inputs (Open-Meteo forecast issued about 24 hours earlier for the hour being "
           "predicted: sunlight, cloud, temperature, direct sun) and a second model that predicts how cloudy it will be "
           "(clear-sky index) instead of raw sunlight; the two models are averaged.")


def split(feat: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS[name])
    tm = feat.index.get_level_values("target_time")
    return feat[(tm >= a) & (tm < b)]


def show(title: str, ev: dict) -> None:
    print(f"\n=== {title}: daytime MAE / RMSE (W/m2 | kW) ===")
    rows = [{"region": "ALL", "model": nzmodel.MODEL_NAMES[k], **m} for k, m in ev["overall"].items()]
    rows += [{"region": r, "model": nzmodel.MODEL_NAMES[k], **m} for r, d in ev["per_region"].items() for k, m in d.items()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def previous_model(old_meta: dict) -> dict:
    """What the model we replace looked like: v1 from a v1 JSON, or the stored 'previous' when re-running v4."""
    if old_meta.get("version") == VERSION:
        return old_meta["previous"]
    return {"version": "v1", "features": len(old_meta["features"]), "chosen_params": old_meta.get("chosen_params"),
            "tuning": old_meta.get("tuning"), "metrics": old_meta["metrics"], "created_utc": old_meta.get("created_utc"),
            "note": "One gradient-boosting model on weather up to now (no forecast inputs)."}


def experiment_chain() -> dict:
    """Summaries of the v2/v3/v4 experiments, read from experiments/results/nz_v*.json (validation period only)."""
    res = {k: json.loads((ROOT / "experiments" / "results" / f"nz_{k}.json").read_text()) for k in ("v2", "v3", "v4")}
    v1 = res["v4"]["v1_val_mae_wm2"]
    c2, c3, c4 = res["v2"]["candidates"], res["v3"]["candidates"], res["v4"]["candidates"]
    return {
        "bar_wm2": MIN_GAIN_WM2, "v1_val_mae_wm2": v1,
        "v2": {"idea": "train on 30 sites instead of the 14 dashboard regions", "result": res["v2"],
               "summary": f"More sites: no real gain ({c2['v1 config, 30 sites']['val_mae_wm2']:.2f} vs "
                          f"{c2['v1 config, 14 sites']['val_mae_wm2']:.2f} W/m2 with the v1 settings; best {res['v2']['best']} "
                          f"{c2[res['v2']['best']]['val_mae_wm2']:.2f}). Not promoted."},
        "v3": {"idea": "add the day-ahead weather forecast for the target hour", "result": res["v3"],
               "summary": f"Forecast inputs: a small gain ({res['v3']['candidates'][res['v3']['best']]['val_mae_wm2']:.2f} vs "
                          f"{v1:.2f} W/m2, gain {res['v3']['gain_wm2']:.2f}), below the {MIN_GAIN_WM2:g} W/m2 bar. Not promoted alone."},
        "v4": {"idea": "predict the clear-sky index too and average it with the raw model, with forecast inputs",
               "result": res["v4"],
               "summary": f"Clear-sky index + averaging: {c4[res['v4']['best']]['val_mae_wm2']:.2f} vs {v1:.2f} W/m2, "
                          f"gain {res['v4']['gain_wm2']:.2f} (> {MIN_GAIN_WM2:g} W/m2 bar). Promoted."},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=(date.today() - timedelta(days=6)).isoformat(), help="last history day (archive lags ~5 days)")
    ap.add_argument("--force", action="store_true", help="save even if the validation gain is below the bar")
    args = ap.parse_args()
    end = args.end

    long_df = data.fetch_history(config.HISTORY_START, end)          # weather + day-ahead forecasts (from 2024)
    print("history:", long_df.shape, long_df["time_utc"].min(), "->", long_df["time_utc"].max())
    feat = build_features(long_df)
    train, val, test = split(feat, "train"), split(feat, "val"), split(feat, "test")
    print("rows train/val/test:", len(train), len(val), len(test))

    old_meta = nzmodel.load_meta()
    previous = previous_model(old_meta)
    model = nzmodel.train(train)                                      # V1_PARAMS, train period only
    val_ev = nzmodel.evaluate(val, nzmodel.all_predictions(model, val))
    show("VALIDATION", val_ev)
    o = val_ev["overall"]
    assert o["model"]["mae_wm2"] < min(o["persistence"]["mae_wm2"], o["clear_sky"]["mae_wm2"]), "model must beat both baselines on val"
    gain = previous["metrics"]["val"]["overall"]["model"]["mae_wm2"] - o["model"]["mae_wm2"]
    print(f"\nval MAE {o['model']['mae_wm2']:.2f} W/m2 vs previous {previous['version']} "
          f"{previous['metrics']['val']['overall']['model']['mae_wm2']:.2f}: gain {gain:.2f} (bar {MIN_GAIN_WM2})")
    if gain <= MIN_GAIN_WM2 and not args.force:
        raise SystemExit("Validation gain is below the bar: not saving and not looking at the test period.")

    test_ev = nzmodel.evaluate(test, nzmodel.all_predictions(model, test))     # the ONE look at test
    show("TEST (reported once)", test_ev)

    report = nzmodel.build_report(model, val, test)
    nzmodel.save(model, {"val": val_ev, "test": test_ev},
                 extra={"version": VERSION, "changes": CHANGES, "chosen_params": nzmodel.V1_PARAMS, "data_end": end,
                        "report": report, "previous": previous, "experiments": experiment_chain(),
                        "learning_curve": old_meta.get("learning_curve"),
                        "learning_curve_note": "Measured for v1 (weather history only); v4 was not re-run for every size.",
                        "data_source": "Open-Meteo (ERA5 reanalysis + archived day-ahead forecasts), CC BY 4.0"})
    print("\nsaved", config.MODEL_PATH, config.MODEL_META)


if __name__ == "__main__":
    main()
