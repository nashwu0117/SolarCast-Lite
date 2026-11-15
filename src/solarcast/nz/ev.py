"""How long does it take to charge an EV from your own solar panels? A simple hour-by-hour estimate.

For the rest of the local day, each forecast hour gives the panels' power (already scaled to the user's system). The home
uses a small base load first; whatever is left goes into the car, up to what the charger and the car's on-board AC
charger accept. An EV charger cannot start below about 1.4 kW (6 A at 230 V), so smaller surpluses are not used.
About 10% is lost between the wall and the battery.

The cars are common in New Zealand: the Tesla Model Y and BYD Atto 3 were the two best-selling new EVs in 2025, and the
Nissan Leaf is the most common used import. Specs are typical published figures (battery size, the most AC power the
car's on-board charger accepts, energy use); real cars and chargers vary, so the result is an estimate, not a promise.
"""
from __future__ import annotations

from solarcast.nz import config

CARS = {
    "tesla_model_y": {"name": "Tesla Model Y", "battery_kwh": 60.0, "ac_max_kw": 11.0, "kwh_per_100km": 15.0,
                      "note": "NZ's best-selling new EV in 2025"},
    "byd_atto_3": {"name": "BYD Atto 3", "battery_kwh": 60.48, "ac_max_kw": 7.0, "kwh_per_100km": 16.0,
                   "note": "No. 2 new EV in NZ in 2025"},
    "tesla_model_3": {"name": "Tesla Model 3", "battery_kwh": 57.5, "ac_max_kw": 11.0, "kwh_per_100km": 13.5, "note": ""},
    "byd_dolphin": {"name": "BYD Dolphin", "battery_kwh": 60.4, "ac_max_kw": 11.0, "kwh_per_100km": 15.0, "note": ""},
    "mg4": {"name": "MG4", "battery_kwh": 64.0, "ac_max_kw": 6.6, "kwh_per_100km": 16.0, "note": ""},
    "kia_ev6": {"name": "Kia EV6", "battery_kwh": 77.4, "ac_max_kw": 11.0, "kwh_per_100km": 17.0, "note": ""},
    "hyundai_kona": {"name": "Hyundai Kona Electric", "battery_kwh": 64.8, "ac_max_kw": 10.8, "kwh_per_100km": 15.0,
                     "note": ""},
    "nissan_leaf": {"name": "Nissan Leaf", "battery_kwh": 40.0, "ac_max_kw": 6.6, "kwh_per_100km": 16.0,
                    "note": "the most common EV on NZ roads, mostly used imports"},
}
HOME_CHARGER_KW = 7.4          # 32 A single-phase wall charger; most NZ homes are single-phase
THREE_PHASE_CHARGER_KW = 11.0  # 16 A three-phase
MIN_CHARGE_KW = 1.4            # 6 A x 230 V: the smallest current an EV charger can run at
EFFICIENCY = 0.9               # wall -> battery
HOME_LOAD_KW = 0.5             # the home's own base load during the day, served first


def plan(hours, solar_kw, next_hour: int, car: str = "tesla_model_y", start_pct: float = 20, target_pct: float = 80,
         charger_kw: float = HOME_CHARGER_KW, home_load_kw: float = HOME_LOAD_KW,
         price_nzd: float = config.GRID_PRICE_NZD) -> dict:
    """Charge `car` from start_pct to target_pct using only solar from local hour `next_hour` to the end of the day.

    hours: local hour of each value (0-23); solar_kw: forecast kW of the user's panels for that hour (None = no data).
    """
    spec = CARS[car]
    start_pct = min(max(float(start_pct), 0.0), 100.0)
    target_pct = min(max(float(target_pct), start_pct), 100.0)
    need = spec["battery_kwh"] * (target_pct - start_pct) / 100          # kWh into the battery
    rate = min(float(charger_kw), spec["ac_max_kw"])                    # kW the car can draw from the wall
    got, finish, charging_h, rows = 0.0, None, 0.0, []
    for h, kw in zip(hours, solar_kw):
        if h < next_hour:
            continue
        kw = max(float(kw or 0.0), 0.0)
        home = min(kw, home_load_kw)
        surplus = kw - home
        draw = min(surplus, rate) if surplus >= MIN_CHARGE_KW else 0.0
        used = 0.0
        if draw > 0 and got < need:
            used = min(draw, (need - got) / EFFICIENCY)                 # stop when the target is reached
            got += used * EFFICIENCY
            charging_h += used / draw
            if got >= need - 1e-9 and finish is None:
                m = round(used / draw * 60)                                 # minutes into this hour
                finish = f"{(h + m // 60) % 24:02d}:{m % 60:02d}"
        rows.append({"hour": int(h), "solar_kw": round(kw, 2), "home_kw": round(home, 2), "car_kw": round(used, 2),
                     "unused_kw": round(kw - home - used, 2)})
    end_pct = start_pct + got / spec["battery_kwh"] * 100
    left = max(need - got, 0.0)
    grid_h = need / EFFICIENCY / rate if rate else None
    return {
        "car": car, "car_name": spec["name"], "battery_kwh": spec["battery_kwh"], "charger_kw": round(rate, 1),
        "start_pct": round(start_pct), "target_pct": round(target_pct), "need_kwh": round(need, 1),
        "done": need == 0 or finish is not None, "finish_local_time": finish,
        "solar_kwh_to_battery": round(got, 1), "end_pct": round(end_pct), "charging_hours": round(charging_h, 1),
        "km_added": round(got / spec["kwh_per_100km"] * 100), "km_target": round(need / spec["kwh_per_100km"] * 100),
        "still_needed_kwh": round(left, 1), "still_needed_grid_cost_nzd": round(left / EFFICIENCY * price_nzd, 2),
        "saved_nzd": round(got / EFFICIENCY * price_nzd, 2),
        "grid_only_hours": None if grid_h is None else round(grid_h, 1),
        "grid_only_cost_nzd": round(need / EFFICIENCY * price_nzd, 2),
        "hours": rows,
    }
