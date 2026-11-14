"""Train SolarCast NZ: fetch history, build features, train on TRAIN, choose settings on VAL, report TEST once.

Usage (run on Colab):  PYTHONPATH=src python scripts/train_nz.py [--end YYYY-MM-DD]
Writes models/nz_model.joblib and models/nz_model.json (metrics + dashboard report).
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarcast.nz import config, data, model as nzmodel          # noqa: E402
from solarcast.nz.features import build_features                 # noqa: E402

# A small grid (12 configs), judged on the validation period only.
GRID = [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves, min_samples_leaf=msl, loss=loss)
        for lr, it, leaves, msl, loss in [
            (0.05, 300, 31, 50, "squared_error"), (0.05, 300, 31, 50, "absolute_error"),
            (0.08, 300, 15, 100, "squared_error"), (0.08, 300, 15, 100, "absolute_error"),
            (0.05, 600, 31, 50, "squared_error"), (0.05, 600, 15, 100, "squared_error"),
            (0.03, 800, 63, 100, "squared_error"), (0.03, 800, 63, 100, "absolute_error"),
            (0.1, 200, 63, 200, "squared_error"), (0.1, 200, 31, 20, "squared_error"),
            (0.05, 400, 63, 50, "squared_error"), (0.05, 400, 31, 200, "absolute_error")]]


def split(feat: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS[name])
    tm = feat.index.get_level_values("target_time")
    return feat[(tm >= a) & (tm < b)]


def show(title: str, ev: dict) -> None:
    print(f"\n=== {title}: daytime MAE / RMSE (W/m2 | kW) ===")
    rows = [{"region": "ALL", "model": nzmodel.MODEL_NAMES[k], **m} for k, m in ev["overall"].items()]
    rows += [{"region": r, "model": nzmodel.MODEL_NAMES[k], **m} for r, d in ev["per_region"].items() for k, m in d.items()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=(date.today() - timedelta(days=6)).isoformat(), help="last history day (archive lags ~5 days)")
    end = ap.parse_args().end

    long_df = data.fetch_history(config.HISTORY_START, end)
    print("history:", long_df.shape, long_df["time_utc"].min(), "->", long_df["time_utc"].max())
    feat = build_features(long_df)
    train, val, test = split(feat, "train"), split(feat, "val"), split(feat, "test")
    print("rows train/val/test:", len(train), len(val), len(test))

    base = {"persistence": nzmodel.persistence(val), "clear_sky": nzmodel.clear_sky_persistence(val)}

    best, tuning = None, []
    for params in GRID:                                        # tune on VAL only
        m = nzmodel.train(train, **params)
        ev = nzmodel.evaluate(val, {"model": nzmodel.predict(m, val), **base})
        mae = ev["overall"]["model"]["mae_wm2"]
        tuning.append({**params, "val_mae_wm2": mae})
        print(f"{params} -> val MAE {mae:.2f} W/m2")
        if best is None or mae < best[0]:
            best = (mae, params, m, ev)
    _, params, model, val_ev = best
    show("VALIDATION (chosen config)", val_ev)
    o = val_ev["overall"]
    assert o["model"]["mae_wm2"] < min(o["persistence"]["mae_wm2"], o["clear_sky"]["mae_wm2"]), "model must beat both baselines on val"

    test_ev = nzmodel.evaluate(test, nzmodel.all_predictions(model, test))     # the ONE look at test
    show("TEST (reported once)", test_ev)

    report = nzmodel.build_report(model, val, test)
    nzmodel.save(model, {"val": val_ev, "test": test_ev},
                 extra={"chosen_params": params, "tuning": tuning, "data_end": end, "report": report,
                        "data_source": "Open-Meteo (ERA5 reanalysis), CC BY 4.0"})
    print("\nchosen:", params, "\nsaved", config.MODEL_PATH, config.MODEL_META)


if __name__ == "__main__":
    main()
