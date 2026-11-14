"""Why does persistence make better charging decisions than Model B on the test period?

Per local hour of the working day, compare forecast bias (forecast - actual), the error of the
expected surplus, and where each strategy charges and imports grid power. Site A.
    PYTHONPATH=src python experiments/charging_diagnosis.py
"""
import numpy as np
import pandas as pd

from solarcast import charging
from solarcast.pipeline import run_experiment

pd.set_option("display.width", 160)
res = run_experiment("A")

for split in ("val", "test"):
    feat = res.splits[split]
    slots = charging.daily_slots(feat)
    actual = slots["actual_kw"]
    print(f"\n=== Site A {split}: forecast bias (forecast - actual, kW) by local hour ===")
    bias = pd.DataFrame({name: (res.predictions[split][name].reindex(slots.index) - actual)
                         for name in ("Persistence", "Model B")})
    bias["hour"] = slots["hour"]
    bias["actual_mean"] = actual
    print(bias.groupby("hour").mean().round(2).to_string())

    # Replay the rule hour by hour to see where charging and grid import happen.
    rows = []
    for name, fc_all in [("Persistence", res.predictions[split]["Persistence"]),
                         ("Model B", res.predictions[split]["Model B"]),
                         ("Perfect", feat["target"])]:
        for date, day in slots.groupby("date"):
            fc = fc_all.reindex(day.index).to_numpy()
            act = day["actual_kw"].to_numpy()
            need, n = charging.EV_NEED_KWH, len(day)
            for i, h in enumerate(day["hour"]):
                must = max(0.0, need - charging.CHARGER_KW * (n - i - 1))
                c = float(np.clip(max(must, fc[i] - charging.WORKSHOP_LOAD_KW), 0, min(charging.CHARGER_KW, need)))
                need -= c
                extra = max(0, charging.WORKSHOP_LOAD_KW + c - act[i]) - max(0, charging.WORKSHOP_LOAD_KW - act[i])
                rows.append({"strategy": name, "hour": h, "charge_kw": c, "ev_grid_kw": extra,
                             "forced": c > 0 and must >= c - 1e-9 and must > 0})
    r = pd.DataFrame(rows)
    print(f"\n--- {split}: mean charge kW / EV grid import kW / share of forced hours, by hour ---")
    print(r.pivot_table(index="hour", columns="strategy", values=["charge_kw", "ev_grid_kw"], aggfunc="mean")
          .round(2).to_string())
    print("EV grid kWh per day:", (r.groupby("strategy")["ev_grid_kw"].sum() / slots["date"].nunique()).round(2).to_dict())
    print("hours where charging was forced by the deadline:", r[r["forced"]].groupby("strategy").size().to_dict())
