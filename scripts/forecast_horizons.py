"""How many hours ahead does SolarCast stay clearly better than the weather service's own forecast?

For each horizon h (the target hour ends h hours after the issue hour), train the same model as v4 (the SolarEnsemble with
the v4 inputs and settings) on 2022-2024 and compare, on the same daytime rows, with:
  * the weather service's day-ahead forecast alone (Open-Meteo, issued about 24 h before the target hour),
  * clear-sky persistence (keep today's cloudiness, move the sun), and
  * persistence (the same sunlight as now).

The rule is fixed in advance and judged on the VALIDATION period (Jan-Jun 2025): a horizon goes into the dashboard's
Forecast mode only if SolarCast's average error is at least RULE_PCT % lower than the weather service's forecast. The
test period (Jul 2025 - Oct 2026) is reported once, for the record, and is not used to choose anything.

Usage (Colab):  PYTHONPATH=src python scripts/forecast_horizons.py  -> experiments/results/nz_horizons.{json,md}
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarcast.nz import config, data, model as nzmodel                 # noqa: E402
from solarcast.nz.ensemble import SolarEnsemble                          # noqa: E402
from solarcast.nz.features import FEATURES, build_features               # noqa: E402

HORIZONS = [1, 2, 3, 6, 12, 24]
RULE_PCT = 10.0
NAMES = {"model": "SolarCast", "weather_service": "Weather service forecast alone",
         "clear_sky": "Clear-sky persistence", "persistence": "Persistence"}


def split(feat: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS[name])
    tm = feat.index.get_level_values("target_time")
    return feat[(tm >= a) & (tm < b)]


def scores(model, part: pd.DataFrame) -> dict:
    """Daytime errors of every method on the SAME rows (rows where the weather-service forecast exists)."""
    keep = part[nzmodel.daytime_mask(part) & part["target_ghi"].notna() & part["fc_ghi_target"].notna()]
    preds = {"model": pd.Series(model.predict(keep), index=keep.index),
             "weather_service": keep["fc_ghi_target"].clip(lower=0),
             "clear_sky": nzmodel.clear_sky_persistence(keep), "persistence": nzmodel.persistence(keep)}
    ev = nzmodel.evaluate(keep, preds)["overall"]
    out = {k: {"mae_kw": round(m["mae_kw"], 3), "mae_wm2": round(m["mae_wm2"], 2),
               "pct_of_capacity": round(m["mae_kw"] / config.SYSTEM_KW * 100, 2)} for k, m in ev.items()}
    out["rows"] = int(len(keep))
    out["better_than_weather_service_pct"] = round((1 - ev["model"]["mae_kw"] / ev["weather_service"]["mae_kw"]) * 100, 1)
    return out


def main():
    meta = nzmodel.load_meta()
    long_df = data.fetch_history(config.HISTORY_START, meta.get("data_end", "2026-11-08"))
    results = {}
    for h in HORIZONS:
        t0 = time.time()
        feat = build_features(long_df, horizon=h)
        model = SolarEnsemble(FEATURES, nzmodel.V1_PARAMS).fit(split(feat, "train"))
        results[h] = {"val": scores(model, split(feat, "val")), "test": scores(model, split(feat, "test")),
                      "seconds": round(time.time() - t0)}
        v, t = results[h]["val"], results[h]["test"]
        print(f"{h:>2} h ahead: val SolarCast {v['model']['pct_of_capacity']}% vs weather service "
              f"{v['weather_service']['pct_of_capacity']}% (better by {v['better_than_weather_service_pct']}%) | "
              f"test {t['model']['pct_of_capacity']}% vs {t['weather_service']['pct_of_capacity']}% "
              f"({t['better_than_weather_service_pct']}%) | {results[h]['seconds']} s", flush=True)
    chosen = [h for h in HORIZONS if results[h]["val"]["better_than_weather_service_pct"] >= RULE_PCT]
    out = {"rule": f"offer a horizon only if SolarCast is >= {RULE_PCT}% better than the weather service's forecast "
                   "on the validation period", "horizons_offered": chosen, "results": results,
           "system_kw": config.SYSTEM_KW, "test_used_for_choice": False}
    res = ROOT / "experiments" / "results"
    (res / "nz_horizons.json").write_text(json.dumps(out, indent=2))
    md = ["# How many hours ahead is SolarCast still better? (daytime, average error as % of panel capacity)", "",
          f"Rule fixed in advance (validation period): offer a horizon only if SolarCast is at least {RULE_PCT:g}% "
          "better than the weather service's own forecast. Test period shown for the record only.", "",
          "| Hours ahead | SolarCast (val) | Weather service (val) | Better by (val) | SolarCast (test) | "
          "Weather service (test) | Better by (test) | Clear-sky persistence (test) | Persistence (test) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        v, t = results[h]["val"], results[h]["test"]
        md.append(f"| {h} | {v['model']['pct_of_capacity']}% | {v['weather_service']['pct_of_capacity']}% | "
                  f"{v['better_than_weather_service_pct']}% | {t['model']['pct_of_capacity']}% | "
                  f"{t['weather_service']['pct_of_capacity']}% | {t['better_than_weather_service_pct']}% | "
                  f"{t['clear_sky']['pct_of_capacity']}% | {t['persistence']['pct_of_capacity']}% |")
    md += ["", f"Offered in the dashboard's Forecast mode: {', '.join(f'{h} h' for h in chosen) or 'none'}."]
    (res / "nz_horizons.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
