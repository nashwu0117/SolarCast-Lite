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


def test_catalog_is_grouped_by_brand_and_complete():
    names = [c["name"] for c in ev.CARS.values()]
    for want in ("Tesla Model S", "Tesla Model X", "Model Y Long Range", "Model 3 Performance", "BYD Sealion 7", "Nissan Leaf"):
        assert any(want in n for n in names), want
    assert all(c["name"].startswith(c["brand"]) for c in ev.CARS.values())
    assert all(k in ev.CARS for k in ev.POPULAR)
    assert "no longer sold new" in ev.CARS["tesla_model_s"]["note"]


def test_byd_plug_in_hybrids_charge_like_evs():
    for k in ("byd_sealion_6", "byd_sealion_5", "byd_sealion_5_std", "byd_shark_6"):
        assert ev.CARS[k]["brand"] == "BYD" and "plug-in hybrid" in ev.CARS[k]["note"]
    assert ev.CARS["byd_shark_6"]["battery_kwh"] == 29.58


def test_home_battery_stores_left_over_sun_and_tops_up_the_car():
    p = ev.plan(HOURS, SUNNY, 8, "tesla_model_y", 20, 80, home_battery="tesla_powerwall_3")
    hb = p["home_battery"]
    left = sum(max(r["unused_kw"], 0) for r in p["hours"])
    assert 0 < hb["stored_kwh"] <= min(left, 13.5) + 0.05
    assert hb["car_end_pct"] >= p["end_pct"]                               # the car was not full, so it gets some
    big = ev.plan(HOURS, [v * 4 for v in SUNNY], 8, "tesla_model_y", 20, 80, home_battery="tesla_powerwall_3")
    hb = big["home_battery"]
    assert hb["stored_kwh"] == 13.5 and hb["full_by"] is not None and hb["to_car_kwh"] == 0   # car done, battery full
    assert hb["home_hours"] == pytest.approx(13.5 * 0.9, abs=0.1)
    assert "home_battery" not in ev.plan(HOURS, SUNNY, 8)


def test_ev_plan_route_with_battery():
    pytest.importorskip("flask")
    from test_webapp import _fake
    from solarcast.nz.webapp import create_app
    c = create_app(_fake.FakeService()).test_client()
    assert "tesla_powerwall_3" in c.get("/api/times").get_json()["home_batteries"]
    r = c.post("/api/ev_plan", json={"hours": HOURS, "solar_kw": SUNNY, "next_hour": 8, "home_battery": "tesla_powerwall_3"})
    assert r.status_code == 200 and r.get_json()["home_battery"]["name"] == "Tesla Powerwall 3"
    assert c.post("/api/ev_plan", json={"hours": HOURS, "solar_kw": SUNNY, "home_battery": "nope"}).status_code == 400
