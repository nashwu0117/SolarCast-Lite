"""Does a better forecast give a better EV-charging decision? And how should the accuracy be read?

Uses only the committed model and replay data (test period, July 2025 - October 2026). Nothing is trained or tuned.

1. Forecast error (daytime, all 14 regions): SolarCast v4 vs persistence, clear-sky persistence and a new baseline,
   Open-Meteo's own day-ahead forecast used directly. Error as a share of the 50 kW capacity AND of the mean daytime
   output, clear vs cloudy hours, and regions that beat clear-sky persistence.
2. Decision value: for every region and local day with all plug-in hours, the same charging rule
   (service.plan_charging) is driven by different forecasts; the grid energy the EV causes is settled with the ACTUAL
   power. Strategies: charge on arrival, a fixed 10:00-14:00 window, persistence, clear-sky, SolarCast v4, and a
   perfect forecast (the upper bound).

Run:  python experiments/nz_decision_value.py   -> experiments/results/nz_decision_value.{json,md}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarcast.nz import app, config, model as nzmodel  # noqa: E402
from solarcast.nz.features import build_features, daytime_mask  # noqa: E402
from solarcast.nz.service import plan_charging  # noqa: E402

OUT = ROOT / "experiments" / "results"
CLOUDY_PCT = 60
FIXED_WINDOW = (10, 14)


def main():
    model = nzmodel.load()
    feat = build_features(app.load_replay())
    t = feat.index.get_level_values("target_time")
    t0, t1 = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS["test"])
    feat = feat[(t >= t0) & (t < t1)]
    preds = {"model": nzmodel.predict(model, feat), "persistence": nzmodel.persistence(feat),
             "clear_sky": nzmodel.clear_sky_persistence(feat),
             "day_ahead_forecast": feat["fc_ghi_target"].fillna(0).clip(lower=0).where(feat["elev_target"] > 0, 0.0)}
    ev = nzmodel.evaluate(feat, preds)
    day = feat[daytime_mask(feat) & feat["target_ghi"].notna()]
    mean_kw = float(day["target_kw"].mean())
    res = {"test_period": config.SPLITS["test"], "daytime_rows": int(len(day)), "mean_daytime_output_kw": round(mean_kw, 2),
           "errors": {k: {"mae_wm2": round(m["mae_wm2"], 2), "mae_kw": round(m["mae_kw"], 3),
                          "pct_of_capacity": round(m["mae_kw"] / config.SYSTEM_KW * 100, 2),
                          "pct_of_mean_output": round(m["mae_kw"] / mean_kw * 100, 1)}
                      for k, m in ev["overall"].items()}}
    e = res["errors"]
    res["improvement_pct"] = {b: round((1 - e["model"]["mae_kw"] / e[b]["mae_kw"]) * 100, 1)
                              for b in ("persistence", "clear_sky", "day_ahead_forecast")}
    res["regions_beating_clear_sky"] = sum(d["model"]["mae_kw"] < d["clear_sky"]["mae_kw"] for d in ev["per_region"].values())
    err = (day["target_ghi"] - preds["model"].loc[day.index]).abs()
    cloudy = day["meta_cloud_now"] > CLOUDY_PCT
    res["mae_wm2_by_sky"] = {f"cloud_le_{CLOUDY_PCT}": round(float(err[~cloudy].mean()), 1),
                             f"cloud_gt_{CLOUDY_PCT}": round(float(err[cloudy].mean()), 1)}

    # ----- decision value -----
    kw = {k: pd.Series(nzmodel.to_kw(v.to_numpy(), feat["temp_t"].to_numpy()), index=feat.index)
          for k, v in preds.items() if k != "day_ahead_forecast"}
    start = feat.index.get_level_values("target_time") - pd.Timedelta("1h")
    local = start.tz_convert(config.LOCAL_TZ)
    frame = pd.DataFrame({"region": feat.index.get_level_values("region"), "date": local.date, "hour": local.hour,
                          "actual": feat["target_kw"].to_numpy(), **{k: v.to_numpy() for k, v in kw.items()}})
    frame = frame[(frame["hour"] >= config.WORK_START) & (frame["hour"] < config.WORK_END)]
    n_slots = config.WORK_END - config.WORK_START
    big = config.WORKSHOP_LOAD_KW + config.CHARGER_KW
    totals = {k: [] for k in ("arrival", "fixed_10_14", "persistence", "clear_sky", "model", "perfect")}
    for _, g in frame.groupby(["region", "date"]):
        if len(g) != n_slots or g["actual"].isna().any():
            continue
        g = g.sort_values("hour")
        act = g["actual"].to_numpy()
        fixed = np.where((g["hour"] >= FIXED_WINDOW[0]) & (g["hour"] < FIXED_WINDOW[1]), big, 0.0)
        drivers = {"arrival": np.full(n_slots, big), "fixed_10_14": fixed, "persistence": g["persistence"].to_numpy(),
                   "clear_sky": g["clear_sky"].to_numpy(), "model": g["model"].to_numpy(), "perfect": act}
        for k, fc in drivers.items():
            totals[k].append(plan_charging(np.nan_to_num(fc), act)[1])
    grid = {k: round(float(np.mean(v)), 2) for k, v in totals.items()}
    res["ev"] = {"region_days": len(totals["model"]), "grid_kwh_per_day": grid,
                 "saved_vs_arrival_kwh_per_day": {k: round(grid["arrival"] - v, 2) for k, v in grid.items() if k != "arrival"},
                 "model_vs_fixed_10_14": {"kwh_per_day": round(grid["fixed_10_14"] - grid["model"], 2),
                                          "nzd_per_day": round((grid["fixed_10_14"] - grid["model"]) * config.GRID_PRICE_NZD, 2),
                                          "kg_co2_per_day": round((grid["fixed_10_14"] - grid["model"]) * config.GRID_CO2_KG_PER_KWH, 2)},
                 "gap_to_perfect_kwh_per_day": round(grid["model"] - grid["perfect"], 2)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nz_decision_value.json").write_text(json.dumps(res, indent=2))
    names = {"model": "SolarCast v4", "persistence": "Persistence", "clear_sky": "Clear-sky persistence",
             "day_ahead_forecast": "Open-Meteo day-ahead forecast only"}
    md = ["# Forecast error and EV decision value (test period, committed model)", "",
          f"Daytime rows: {res['daytime_rows']}; mean daytime output {mean_kw:.2f} kW of 50 kW.", "",
          "| Method | MAE W/m2 | MAE kW | % of 50 kW capacity | % of mean daytime output |", "|---|---|---|---|---|"]
    md += [f"| {names[k]} | {v['mae_wm2']} | {v['mae_kw']} | {v['pct_of_capacity']} | {v['pct_of_mean_output']} |" for k, v in e.items()]
    md += ["", f"Improvement of v4: {res['improvement_pct']}. Regions beating clear-sky persistence: "
               f"{res['regions_beating_clear_sky']}/14. MAE by sky: {res['mae_wm2_by_sky']} W/m2.", "",
           f"## EV charging ({res['ev']['region_days']} region-days, grid energy caused by the EV, kWh/day)", "",
           "| Strategy | Grid kWh/day |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in grid.items()]
    md += ["", f"v4 vs fixed 10:00-14:00 window: {res['ev']['model_vs_fixed_10_14']}; gap to a perfect forecast: "
               f"{res['ev']['gap_to_perfect_kwh_per_day']} kWh/day."]
    (OUT / "nz_decision_value.md").write_text("\n".join(md) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
