"""Charging an EV from the user's own panels: the hour-by-hour estimate and its route."""
import pytest

from solarcast.nz import ev

HOURS = list(range(24))
# a 5 kW system on a clear day: 0 at night, up to 4.5 kW at noon
SUNNY = [0, 0, 0, 0, 0, 0, 0.2, 0.8, 1.8, 2.9, 3.8, 4.3, 4.5, 4.3, 3.8, 2.9, 1.8, 0.8, 0.2, 0, 0, 0, 0, 0]


def test_big_system_finishes_and_reports_a_time():
    big = [v * 4 for v in SUNNY]                                          # 20 kW of panels
    p = ev.plan(HOURS, big, 8, "tesla_model_y", 20, 80)
    assert p["done"] and p["need_kwh"] == 36.0 and p["finish_local_time"] is not None
    assert p["solar_kwh_to_battery"] == pytest.approx(36.0, abs=0.05) and p["end_pct"] == 80
    assert all(r["car_kw"] <= ev.HOME_CHARGER_KW + 1e-9 for r in p["hours"])   # never above the charger
    assert p["grid_only_hours"] == pytest.approx(36 / 0.9 / 7.4, abs=0.05)


def test_car_limit_and_small_surplus():
    big = [v * 4 for v in SUNNY]
    byd = ev.plan(HOURS, big, 8, "byd_atto_3", 20, 80, charger_kw=ev.THREE_PHASE_CHARGER_KW)
    assert byd["charger_kw"] == 7.0 and max(r["car_kw"] for r in byd["hours"]) <= 7.0     # the car's AC limit
    p = ev.plan(HOURS, SUNNY, 8, "tesla_model_y", 20, 80)
    for r in p["hours"]:
        if r["solar_kw"] - r["home_kw"] < ev.MIN_CHARGE_KW:
            assert r["car_kw"] == 0                                        # too little to start a charger


def test_small_system_gets_part_way_and_says_what_is_left():
    p = ev.plan(HOURS, SUNNY, 8, "tesla_model_y", 20, 80)
    assert not p["done"] and p["finish_local_time"] is None
    assert 20 < p["end_pct"] < 80 and p["still_needed_kwh"] > 0 and p["km_added"] > 0
    assert p["still_needed_kwh"] == pytest.approx(p["need_kwh"] - p["solar_kwh_to_battery"], abs=0.1)


def test_hours_already_past_are_not_used():
    p = ev.plan(HOURS, SUNNY, 19, "tesla_model_y", 20, 80)                # evening: no sun left
    assert p["solar_kwh_to_battery"] == 0 and not p["done"] and all(r["hour"] >= 19 for r in p["hours"])
    assert ev.plan(HOURS, SUNNY, 8, "tesla_model_y", 80, 80)["done"]       # nothing to charge


def test_ev_plan_route():
    pytest.importorskip("flask")
    from test_webapp import _fake
    from solarcast.nz.webapp import create_app
    c = create_app(_fake.FakeService()).test_client()
    assert "byd_atto_3" in c.get("/api/times").get_json()["ev_cars"]
    r = c.post("/api/ev_plan", json={"hours": HOURS, "solar_kw": SUNNY, "next_hour": 8, "car": "byd_atto_3",
                                     "start_pct": 30, "target_pct": 60})
    assert r.status_code == 200 and r.get_json()["car_name"].startswith("BYD")
    assert c.post("/api/ev_plan", json={"hours": HOURS, "solar_kw": SUNNY, "car": "nope"}).status_code == 400
    assert c.post("/api/ev_plan", json={}).status_code == 400
