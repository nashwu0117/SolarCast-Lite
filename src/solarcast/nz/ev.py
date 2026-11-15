"""How long does it take to charge an EV from your own solar panels? A simple hour-by-hour estimate.

For the rest of the local day, each forecast hour gives the panels' power (already scaled to the user's system). The home
uses a small base load first; whatever is left goes into the car, up to what the charger and the car's on-board AC
charger accept. An EV charger cannot start below about 1.4 kW (6 A at 230 V), so smaller surpluses are not used.
About 10% is lost between the wall and the battery.

The cars are the EVs sold in New Zealand, new or as common used imports: the Tesla Model Y and BYD Atto 3 were the two
best-selling new EVs in 2025, and the Nissan Leaf is the most common used import. Tesla dropped the right-hand-drive Model S
and X in 2023, so those are older cars already here. BYD's plug-in hybrids (Sealion 5, Sealion 6, Shark 6) are included too:
they charge from a wall socket like an EV, only with a smaller battery. Specs are typical published figures (usable battery size, the most AC
power the car's on-board charger accepts, energy use); real cars and chargers vary, so the result is an estimate.

A home battery (for example a Tesla Powerwall 3, 13.5 kWh, sold in NZ since 2024) can store the solar that is left over
during the day. In the evening it first tops up the car if the panels could not finish, and the rest runs the home.
"""
from __future__ import annotations

from solarcast.nz import config

_U = "older cars on NZ roads; no longer sold new in NZ"
_P = "plug-in hybrid (PHEV): petrol takes over when the battery is empty"


def _car(brand, model, battery_kwh, ac_max_kw, kwh_per_100km, note=""):
    return {"brand": brand, "name": f"{brand} {model}", "battery_kwh": battery_kwh, "ac_max_kw": ac_max_kw,
            "kwh_per_100km": kwh_per_100km, "note": note}


# EVs sold in New Zealand (new, or common as used imports), grouped by brand. battery_kwh is the usable size.
CARS = {
    # Tesla: the Model Y and Model 3 are sold new; right-hand-drive Model S and X were dropped in 2023.
    "tesla_model_y": _car("Tesla", "Model Y RWD", 60.0, 11.0, 15.0, "NZ's best-selling new EV in 2025"),
    "tesla_model_y_lr": _car("Tesla", "Model Y Long Range", 75.0, 11.0, 16.0),
    "tesla_model_y_perf": _car("Tesla", "Model Y Performance", 75.0, 11.0, 17.5),
    "tesla_model_3": _car("Tesla", "Model 3 RWD", 57.5, 11.0, 13.5),
    "tesla_model_3_lr": _car("Tesla", "Model 3 Long Range", 75.0, 11.0, 14.0),
    "tesla_model_3_perf": _car("Tesla", "Model 3 Performance", 75.0, 11.0, 15.5),
    "tesla_model_s": _car("Tesla", "Model S Long Range", 95.0, 11.0, 18.0, _U),
    "tesla_model_x": _car("Tesla", "Model X Long Range", 95.0, 11.0, 21.0, _U),
    # BYD
    "byd_atto_3": _car("BYD", "Atto 3 Extended", 60.48, 7.0, 16.0, "No. 2 new EV in NZ in 2025"),
    "byd_atto_3_std": _car("BYD", "Atto 3 Standard", 49.92, 7.0, 15.5),
    "byd_atto_2": _car("BYD", "Atto 2", 45.12, 7.0, 14.5),
    "byd_dolphin": _car("BYD", "Dolphin Premium", 60.48, 7.0, 15.0),
    "byd_dolphin_std": _car("BYD", "Dolphin Dynamic", 44.9, 7.0, 14.0),
    "byd_seal": _car("BYD", "Seal", 82.56, 7.0, 16.0),
    "byd_sealion_7": _car("BYD", "Sealion 7", 82.56, 11.0, 19.0),
    # BYD plug-in hybrids ("DM-i super hybrid"): they plug in like an EV and the petrol engine takes over when the battery
    # is empty. Battery sizes are BYD's published pack sizes; energy use is from the WLTP electric range.
    "byd_sealion_6": _car("BYD", "Sealion 6 plug-in hybrid", 18.3, 7.0, 20.0, _P),
    "byd_sealion_5": _car("BYD", "Sealion 5 Premium plug-in hybrid", 18.3, 6.6, 19.5, _P),
    "byd_sealion_5_std": _car("BYD", "Sealion 5 Essential plug-in hybrid", 12.9, 6.6, 19.5, _P),
    "byd_shark_6": _car("BYD", "Shark 6 plug-in hybrid ute", 29.58, 6.6, 26.0, _P),
    # MG
    "mg4": _car("MG", "MG4 Long Range", 64.0, 6.6, 16.0),
    "mg4_std": _car("MG", "MG4 Excite 51", 51.0, 6.6, 16.0),
    "mg4_xpower": _car("MG", "MG4 XPOWER", 64.0, 11.0, 18.0),
    "mg_zs_ev": _car("MG", "ZS EV Long Range", 72.6, 7.0, 17.5),
    # Kia
    "kia_ev3": _car("Kia", "EV3 Long Range", 81.4, 11.0, 15.0),
    "kia_ev5": _car("Kia", "EV5", 88.1, 11.0, 18.0),
    "kia_ev6": _car("Kia", "EV6", 77.4, 11.0, 17.0),
    "kia_niro_ev": _car("Kia", "Niro EV", 64.8, 11.0, 16.0),
    # Hyundai
    "hyundai_kona": _car("Hyundai", "Kona Electric Extended", 64.8, 10.8, 15.0),
    "hyundai_kona_std": _car("Hyundai", "Kona Electric Standard", 48.6, 10.8, 14.5),
    "hyundai_inster": _car("Hyundai", "Inster", 49.0, 11.0, 14.5),
    "hyundai_ioniq_5": _car("Hyundai", "Ioniq 5 Long Range", 84.0, 10.9, 17.0),
    "hyundai_ioniq_6": _car("Hyundai", "Ioniq 6 Long Range", 77.4, 10.9, 14.5),
    # Polestar, Volvo, Toyota, Nissan
    "polestar_2": _car("Polestar", "Polestar 2 Long Range", 79.0, 11.0, 16.5),
    "polestar_4": _car("Polestar", "Polestar 4 Long Range", 94.0, 11.0, 18.5),
    "volvo_ex30": _car("Volvo", "EX30 Extended Range", 64.0, 11.0, 16.0),
    "volvo_ex40": _car("Volvo", "EX40 Twin Motor", 79.0, 11.0, 19.0),
    "toyota_bz4x": _car("Toyota", "bZ4X", 64.0, 6.6, 17.0),
    "nissan_leaf": _car("Nissan", "Leaf 40 kWh", 40.0, 6.6, 16.0, "the most common EV on NZ roads, mostly used imports"),
    "nissan_leaf_62": _car("Nissan", "Leaf e+ 62 kWh", 59.0, 6.6, 17.0),
    "nissan_leaf_24": _car("Nissan", "Leaf 24 kWh (older)", 21.0, 3.3, 15.0, "older used imports"),
}
# The assistant gets plans only for these (all cars would make its facts too long).
POPULAR = ["tesla_model_y", "byd_atto_3", "tesla_model_3", "byd_sealion_7", "mg4", "kia_ev3", "nissan_leaf"]

# Home batteries sold and installed in NZ: usable size and the most power they take from the panels.
HOME_BATTERIES = {
    "tesla_powerwall_3": {"name": "Tesla Powerwall 3", "capacity_kwh": 13.5, "power_kw": 5.0},
    "tesla_powerwall_3_x2": {"name": "Tesla Powerwall 3 + Expansion", "capacity_kwh": 27.0, "power_kw": 5.0},
    "byd_hvs_10": {"name": "BYD Battery-Box Premium HVS 10.2", "capacity_kwh": 10.24, "power_kw": 5.1},
    "sungrow_sbr096": {"name": "Sungrow SBR096", "capacity_kwh": 9.6, "power_kw": 5.0},
    "alpha_smile5": {"name": "AlphaESS SMILE5 10.1", "capacity_kwh": 10.1, "power_kw": 5.0},
}
BATTERY_ROUND_TRIP = 0.9       # energy back out of a home battery for each unit put in
EVENING_LOAD_KW = 1.0          # a typical NZ home in the evening (cooking, heat pump, lights)
HOME_CHARGER_KW = 7.4          # 32 A single-phase wall charger; most NZ homes are single-phase
THREE_PHASE_CHARGER_KW = 11.0  # 16 A three-phase
MIN_CHARGE_KW = 1.4            # 6 A x 230 V: the smallest current an EV charger can run at
EFFICIENCY = 0.9               # wall -> battery
HOME_LOAD_KW = 0.5             # the home's own base load during the day, served first


def plan(hours, solar_kw, next_hour: int, car: str = "tesla_model_y", start_pct: float = 20, target_pct: float = 80,
         charger_kw: float = HOME_CHARGER_KW, home_load_kw: float = HOME_LOAD_KW,
         price_nzd: float = config.GRID_PRICE_NZD, home_battery: str | None = None) -> dict:
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
    out = {
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
    if home_battery:
        out["home_battery"] = store_leftover(rows, home_battery, left, spec["battery_kwh"], end_pct, price_nzd)
    return out


def store_leftover(rows, battery: str, car_left_kwh: float, car_battery_kwh: float, car_end_pct: float,
                   price_nzd: float = config.GRID_PRICE_NZD) -> dict:
    """Put each hour's left-over solar into a home battery (empty in the morning), then use it in the evening:
    first to finish the car, then to run the home instead of buying grid power."""
    b = HOME_BATTERIES[battery]
    stored, full_by = 0.0, None
    for r in rows:
        add = min(max(r["unused_kw"], 0.0), b["power_kw"], b["capacity_kwh"] - stored)
        if add > 0:
            stored += add
            if stored >= b["capacity_kwh"] - 1e-9 and full_by is None:
                m = round(add / min(max(r["unused_kw"], 0.0), b["power_kw"]) * 60)
                full_by = f"{(r['hour'] + m // 60) % 24:02d}:{m % 60:02d}"
    usable = stored * BATTERY_ROUND_TRIP
    to_car = min(usable * EFFICIENCY, car_left_kwh)                       # kWh into the car's battery
    to_home = usable - to_car / EFFICIENCY
    return {"battery": battery, "name": b["name"], "capacity_kwh": b["capacity_kwh"], "stored_kwh": round(stored, 1),
            "full_by": full_by, "to_car_kwh": round(to_car, 1),
            "car_end_pct": round(car_end_pct + to_car / car_battery_kwh * 100),
            "to_home_kwh": round(to_home, 1), "home_hours": round(to_home / EVENING_LOAD_KW, 1),
            "saved_nzd": round(usable * price_nzd, 2)}
