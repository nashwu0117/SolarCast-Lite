"""Print the EV charging comparison for both sites (val and test). Run on Colab:
    PYTHONPATH=src python scripts/run_charging.py
The charging rule and its constants are fixed in solarcast/charging.py; nothing is tuned here.
"""
import pandas as pd

from solarcast import charging
from solarcast.pipeline import run_experiment

pd.set_option("display.width", 160)
COLS = ["days", "ev_grid_kwh", "solar_share_pct", "grid_kwh_saved", "money_saved_eur", "co2_saved_kg", "pct_of_perfect"]

for site in ("A", "B"):
    res = run_experiment(site)
    for split in ("val", "test"):
        preds = {k: v for k, v in res.predictions[split].items() if k != "Previous-day same hour"}
        _, summary = charging.simulate(res.splits[split], preds)
        print(f"\n=== Site {site}, {split} ===")
        print(summary[COLS].to_string())
