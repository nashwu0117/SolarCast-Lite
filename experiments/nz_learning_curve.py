"""[ARCHIVED experiment, written against the v1 model code (one gradient-boosting model, 29 features); the results are
in the matching experiments/results file. nzmodel.train now returns the v4 ensemble, so a rerun needs the v1 code.]
How many days of history give the best model for all of New Zealand?  (validation period only, no test data)

Trains the chosen config on the most recent N days before the val start (14 regions pooled) and scores daytime
MAE on VAL next to the two baselines. Run after scripts/train_nz.py (needs models/nz_model.json and the cached data):
    PYTHONPATH=src python experiments/nz_learning_curve.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from solarcast.nz import config, data, model as nzmodel          # noqa: E402
from solarcast.nz.features import build_features                 # noqa: E402
from train_nz import split                                       # noqa: E402

DAYS = [7, 30, 90, 180, 365, 730, None]          # None = all history before val (about 1096 days)


def main() -> None:
    meta = nzmodel.load_meta()
    params, end = meta["chosen_params"], meta["data_end"]
    feat = build_features(data.fetch_history(config.HISTORY_START, end))
    train, val = split(feat, "train"), split(feat, "val")
    val_start = pd.Timestamp(config.SPLITS["val"][0], tz="UTC")
    tm = train.index.get_level_values("target_time")
    base = {"persistence": nzmodel.persistence(val), "clear_sky": nzmodel.clear_sky_persistence(val)}

    rows = []
    for n in DAYS:
        sub = train if n is None else train[tm >= val_start - pd.Timedelta(days=n)]
        t0 = time.time()
        m = nzmodel.train(sub, **params)
        fit_s = time.time() - t0
        o = nzmodel.evaluate(val, {"model": nzmodel.predict(m, val), **base})["overall"]
        rows.append({"days": "all (~1096)" if n is None else n, "train_rows": len(sub), "fit_s": round(fit_s, 1),
                     "model_mae_wm2": o["model"]["mae_wm2"], "model_mae_kw": o["model"]["mae_kw"],
                     "model_nmae_pct": o["model"]["mae_kw"] / config.SYSTEM_KW * 100,
                     "persistence_mae_wm2": o["persistence"]["mae_wm2"], "clear_sky_mae_wm2": o["clear_sky"]["mae_wm2"]})
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows).round(3)
    md = ("# NZ learning curve (validation period only)\n\nDaytime MAE on val, all 14 regions pooled; chosen config "
          f"`{params}`.\n\n" + df.to_markdown(index=False) + "\n")
    (ROOT / "experiments" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "results" / "nz_learning_curve.md").write_text(md)
    meta["learning_curve"] = df.to_dict("records")
    config.MODEL_META.write_text(json.dumps(meta, indent=2))
    print(md)


if __name__ == "__main__":
    main()
