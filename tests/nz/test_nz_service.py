import json

import numpy as np
import pandas as pd
import pytest

from solarcast.nz import config, data, model as nzmodel, service as svc
from solarcast.nz.features import build_features
from conftest import TEST_REGIONS, make_long


@pytest.fixture(scope="module")
def service(long_df):
    feat = build_features(long_df)
    m = nzmodel.train(feat[feat.index.get_level_values("target_time") < "2025-12-01"], max_iter=40)
    return svc.Service(m, long_df, meta={})


def test_available_and_default_times(service):
    times = service.available_times()
    assert times == sorted(times) and times[0].endswith("Z") and len(times) > 1000
    d = service.default_time()
    assert d in times
    local = pd.Timestamp(d).tz_convert(config.LOCAL_TZ)
    assert local.hour == 12 and local.month in (12, 1, 2)


def test_snapshot_replay(service):
    snap = service.snapshot(service.default_time())
    assert [s["name"] for s in snap] == TEST_REGIONS or {s["name"] for s in snap} == set(TEST_REGIONS)
    keys = {"name", "lat", "lon", "local_time", "cloud_cover_pct", "temp_c", "wind_kmh", "ghi_now_wm2", "is_day",
            "predicted_kw_next", "persistence_kw_next", "actual_kw_next", "generating"}
    for s in snap:
        assert keys <= set(s) and s["actual_kw_next"] is not None
        assert s["generating"] == (s["predicted_kw_next"] > 1)
    json.dumps(snap)


def test_detail_replay(service):
    d = service.detail("Auckland", service.default_time())
    n = len(d["hours"])
    assert n >= 23 and all(len(d[k]) == n for k in ("actual_kw", "predicted_kw", "persistence_kw", "cloud_cover_pct"))
    assert isinstance(d["charge_now"], bool) and isinstance(d["reason"], str) and d["reason"]
    assert {"grid_kwh_saved", "money_saved_nzd", "co2_saved_kg", "metrics"} <= set(d)
    assert d["money_saved_nzd"] == pytest.approx(d["grid_kwh_saved"] * config.GRID_PRICE_NZD, abs=0.01)
    json.dumps(d)


def test_charging_rule():
    charges, grid = svc.plan_charging(np.full(10, 30.0), np.full(10, 30.0))      # plenty of sun: no grid, EV full
    assert sum(charges) == pytest.approx(config.EV_NEED_KWH) and grid == 0
    charges, _ = svc.plan_charging(np.zeros(10), np.zeros(10))                   # no sun: still finishes by 18:00
    assert sum(charges) == pytest.approx(config.EV_NEED_KWH) and charges[0] == 0 and charges[-1] > 0


def test_live_mode(service, monkeypatch):
    now = pd.Timestamp("2026-01-15 03:00", tz="UTC")
    live = make_long("2026-01-13 03:00", "2026-01-16 04:00")
    live = live[live["time_utc"] <= now + pd.Timedelta("24h")]
    live["is_forecast"] = live["time_utc"] > now
    monkeypatch.setattr(data, "fetch_live", lambda regions=None: live)
    service._live = None
    snap = service.snapshot("live")
    assert len(snap) == 3 and all(s["actual_kw_next"] is None for s in snap)
    d = service.detail("Wellington", "live")
    assert d["actual_kw"][-1] is None and d["predicted_kw"][-1] is not None


def test_model_report(service):
    rep = service.model_report()
    assert {"periods", "overall", "per_region", "importance", "sample"} == set(rep)
    assert len(rep["overall"]) == 3 and len(rep["importance"]) > 0
    assert service.model_report() is rep                                         # computed once


def test_model_report_from_saved_meta(long_df):
    saved = {"report": {"periods": {}, "overall": [], "per_region": [], "importance": [], "sample": {}}}
    s = svc.Service(nzmodel.train(build_features(long_df).iloc[:5000], max_iter=5), long_df, meta=saved)
    assert s.model_report() is saved["report"]


def test_model_report_adds_version_and_previous(long_df):
    m = {"mae_kw": 1.0, "mae_wm2": 30.0}
    prev = {"version": "v1", "note": "n", "metrics": {"val": {"overall": {"model": m}}, "test": {"overall": {"model": m}}}}
    meta = {"report": {"periods": {}, "overall": [], "per_region": [], "importance": [], "sample": {}},
            "version": "v4", "changes": "x", "previous": prev, "experiments": {"bar_wm2": 1.0}}
    s = svc.Service(nzmodel.train(build_features(long_df).iloc[:5000], max_iter=5), long_df, meta=meta)
    rep = s.model_report()
    assert rep["version"] == "v4" and rep["previous"]["test"] == m and rep["experiments"] == {"bar_wm2": 1.0}
    json.dumps(rep)


def test_point_anywhere_in_nz(service, monkeypatch):
    """A clicked point downloads its own weather (mocked here) and runs the same model."""
    calls = []

    def fake_fetch_point(lat, lon, name, start=None, end=None):
        calls.append((lat, lon, start, end))
        g = make_long("2025-12-20", "2026-03-01", ["Wellington"], seed=3)
        return g.assign(region=name).drop(columns=[])

    monkeypatch.setattr(svc.data, "fetch_point", fake_fetch_point)
    t = service.default_time()
    d = service.point(-41.0, 175.5, t)                    # Wairarapa, between Wellington and Palmerston North
    assert d["is_point"] and d["nearest_region"] in config.REGIONS and d["distance_km"] > 0
    assert len(d["hours"]) == len(d["predicted_kw"]) == len(d["cloud_cover_pct"]) and "charge_now" in d
    assert d["predicted_kw_next"] is not None and d["cloud_now_pct"] is not None
    json.dumps(d)
    service.point(-41.0, 175.5, t)                        # same point and day: served from the cache
    assert len(calls) == 1
    with pytest.raises(ValueError):
        service.point(-33.0, 151.2, t)                    # Sydney is not in New Zealand


def test_nearest_region():
    assert svc.nearest_region(-43.5, 172.6)[0] == "Christchurch"
    assert svc.nearest_region(-45.0, 168.7)[0] == "Queenstown"
