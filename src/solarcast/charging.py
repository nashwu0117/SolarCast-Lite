"""Smart EV charging: turn the next-hour solar forecast into an hourly charging decision.

Scenario (synthetic assumptions, stated in the write-up): a small workshop with the 50 kW rooftop PV system
of the dataset and one electric vehicle. The EV is plugged in from WORK_START to WORK_END (local time) every
day and must receive EV_NEED_KWH before it leaves. Each hour we choose how much power (0..CHARGER_KW) to give
it. All rules are fixed in advance; nothing is tuned on the test period.

Rule for every strategy (only the forecast differs):
    expected_surplus = forecast_solar - workshop_load
    must_now         = energy still needed that cannot fit into the remaining hours at full power
    charge           = clip(max(must_now, expected_surplus), 0, min(CHARGER_KW, energy still needed))
What it costs is judged with the ACTUAL solar power of that hour:
    grid_import = max(0, workshop_load + charge - actual_solar)
"""
import numpy as np
import pandas as pd

from solarcast import config

WORK_START, WORK_END = 8, 18        # EV plugged in 08:00-18:00 local time (10 hourly slots)
EV_NEED_KWH = 40.0                  # energy the EV needs every day
CHARGER_KW = 11.0                   # typical 3-phase AC wallbox
WORKSHOP_LOAD_KW = 6.0              # workshop machines and lights during opening hours
GRID_PRICE_PER_KWH = 0.25           # EUR per kWh bought from the grid (assumption)
GRID_CO2_KG_PER_KWH = 0.26          # kg CO2 per grid kWh (assumption, order of magnitude for Romania)


def daily_slots(feat: pd.DataFrame) -> pd.DataFrame:
    """Rows of `feat` whose target hour falls in opening hours, tagged with the local date and hour.

    Only days with all opening-hour slots present are kept, so every day is simulated in full.
    """
    local = feat.index.tz_convert(config.LOCAL_TZ)
    slots = pd.DataFrame({"date": local.date, "hour": local.hour, "actual_kw": feat["target"].to_numpy()},
                         index=feat.index)
    slots = slots[(slots["hour"] >= WORK_START) & (slots["hour"] < WORK_END)]
    n_slots = WORK_END - WORK_START
    complete = slots.groupby("date")["hour"].transform("count") == n_slots
    return slots[complete]


def simulate_day(forecast_kw: np.ndarray, actual_kw: np.ndarray) -> dict:
    """Run the charging rule through one day's opening hours; return energy totals in kWh."""
    need = EV_NEED_KWH
    grid_ev = 0.0          # grid energy caused by the EV (import with EV minus import without it)
    n = len(forecast_kw)
    for i in range(n):
        hours_after = n - i - 1
        must_now = max(0.0, need - CHARGER_KW * hours_after)
        surplus = forecast_kw[i] - WORKSHOP_LOAD_KW
        charge = float(np.clip(max(must_now, surplus), 0.0, min(CHARGER_KW, need)))
        need -= charge
        import_with = max(0.0, WORKSHOP_LOAD_KW + charge - actual_kw[i])
        import_without = max(0.0, WORKSHOP_LOAD_KW - actual_kw[i])
        grid_ev += import_with - import_without
    charged = EV_NEED_KWH - need
    return {"ev_kwh": charged, "ev_grid_kwh": grid_ev, "ev_solar_kwh": charged - grid_ev}


def charge_on_arrival(n: int) -> np.ndarray:
    """A 'forecast' that makes the rule charge at full power from the first hour (no planning)."""
    return np.full(n, WORKSHOP_LOAD_KW + CHARGER_KW)


def simulate(feat: pd.DataFrame, predictions: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare charging strategies over every complete day in `feat`.

    predictions: {name: Series of forecast kW indexed like feat}. Two strategies are added automatically:
    "Charge on arrival" (no forecast) and "Perfect forecast" (uses the actual power, an upper bound).
    Returns (daily results in long format, summary table with one row per strategy).
    """
    slots = daily_slots(feat)
    strategies = {"Charge on arrival": None, **predictions, "Perfect forecast": feat["target"]}
    rows = []
    for date, day in slots.groupby("date"):
        actual = day["actual_kw"].to_numpy()
        for name, series in strategies.items():
            fc = charge_on_arrival(len(day)) if series is None else series.reindex(day.index).to_numpy()
            if np.isnan(fc).any():
                continue
            rows.append({"date": date, "strategy": name, **simulate_day(fc, actual)})
    daily = pd.DataFrame(rows)
    summary = summarize(daily)
    return daily, summary


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    """Totals per strategy, with money and CO2 saved compared to charging on arrival."""
    s = daily.groupby("strategy")[["ev_kwh", "ev_grid_kwh", "ev_solar_kwh"]].sum()
    s["days"] = daily.groupby("strategy")["date"].nunique()
    s["solar_share_pct"] = 100 * s["ev_solar_kwh"] / s["ev_kwh"]
    base = s.loc["Charge on arrival", "ev_grid_kwh"] if "Charge on arrival" in s.index else np.nan
    s["grid_kwh_saved"] = base - s["ev_grid_kwh"]
    s["money_saved_eur"] = s["grid_kwh_saved"] * GRID_PRICE_PER_KWH
    s["co2_saved_kg"] = s["grid_kwh_saved"] * GRID_CO2_KG_PER_KWH
    best = s.loc["Perfect forecast", "grid_kwh_saved"] if "Perfect forecast" in s.index else np.nan
    s["pct_of_perfect"] = 100 * s["grid_kwh_saved"] / best
    return s.sort_values("ev_grid_kwh").round(1)
