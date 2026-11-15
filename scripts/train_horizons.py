"""Train the 2- and 3-hours-ahead models for the dashboard's "Predict next" mode and save them next to the main model.

scripts/forecast_horizons.py showed (experiments/results/nz_horizons.md) that on the unseen test period SolarCast beats
the weather service's own forecast clearly at 1 h (40 %) and 2 h (16 %), a little at 3 h (6 %) and not at all from 6 h.
So the dashboard predicts up to 3 hours ahead. Same model, inputs and training period (2022-2024) as that experiment.

Usage (Colab):  PYTHONPATH=src python scripts/train_horizons.py  -> models/nz_model_h2.joblib, models/nz_model_h3.joblib
"""
import json
import sys
from pathlib import Path

import joblib
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarcast.nz import config, data, model as nzmodel                 # noqa: E402
from solarcast.nz.ensemble import SolarEnsemble                          # noqa: E402
from solarcast.nz.features import FEATURES, build_features               # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from forecast_horizons import split                                      # noqa: E402

AHEAD = [2, 3]


def main():
    meta = nzmodel.load_meta()
    long_df = data.fetch_history(config.HISTORY_START, meta.get("data_end", "2026-11-08"))
    for h in AHEAD:
        model = SolarEnsemble(FEATURES, nzmodel.V1_PARAMS).fit(split(build_features(long_df, horizon=h), "train"))
        joblib.dump(model, ROOT / "models" / f"nz_model_h{h}.joblib", compress=3)
        print(f"saved the {h} h model", flush=True)
    res = json.loads((ROOT / "experiments" / "results" / "nz_horizons.json").read_text())["results"]
    info = {"sklearn_version": sklearn.__version__, "features": FEATURES, "trained_on": config.SPLITS["train"],
            "source": "experiments/results/nz_horizons.md",
            "test": {str(h): {"pct_of_capacity": res[str(h)]["test"]["model"]["pct_of_capacity"],
                              "weather_service_pct_of_capacity": res[str(h)]["test"]["weather_service"]["pct_of_capacity"],
                              "better_than_weather_service_pct": res[str(h)]["test"]["better_than_weather_service_pct"]}
                     for h in [1] + AHEAD}}
    (ROOT / "models" / "nz_horizons.json").write_text(json.dumps(info, indent=2))
    print("TRAINDONE", flush=True)


if __name__ == "__main__":
    main()
