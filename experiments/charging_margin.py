"""Pick the charging safety margin on the VALIDATION period only, then report the test period once.
    PYTHONPATH=src python experiments/charging_margin.py
"""
import pandas as pd

from solarcast import charging
from solarcast.pipeline import run_experiment

pd.set_option("display.width", 160)
MARGINS = [0, 2, 4, 6, 8, 10]
results = {s: run_experiment(s) for s in ("A", "B")}


def preds(res, split):
    return {k: v for k, v in res.predictions[split].items() if k in ("Persistence", "Model A", "Model B")}


# 1) Validation: EV grid kWh per strategy and margin, summed over both sites.
rows = []
for m in MARGINS:
    for site, res in results.items():
        _, s = charging.simulate(res.splits["val"], preds(res, "val"), margin=m)
        for strat, kwh in s["ev_grid_kwh"].items():
            rows.append({"margin_kw": m, "site": site, "strategy": strat, "ev_grid_kwh": kwh})
val = pd.DataFrame(rows).pivot_table(index="margin_kw", columns="strategy", values="ev_grid_kwh", aggfunc="sum")
print("=== VALIDATION (Jan-Feb 2026), EV grid kWh, Site A + B ===")
print(val.round(1).to_string())
best = int(val["Model B"].idxmin())
print(f"\nMargin chosen on validation for Model B: {best} kW")

# 2) Test, reported once with the chosen margin (and margin 0 for reference).
for m in sorted({0, best}):
    for site, res in results.items():
        _, s = charging.simulate(res.splits["test"], preds(res, "test"), margin=m)
        print(f"\n=== TEST (Mar-Jun 2026), Site {site}, margin {m} kW ===")
        print(s[["days", "ev_grid_kwh", "solar_share_pct", "grid_kwh_saved", "money_saved_eur",
                 "co2_saved_kg", "pct_of_perfect"]].to_string())
