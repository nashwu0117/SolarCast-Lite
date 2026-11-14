"""Nationwide grid: cells, national numbers, replay/live snapshots, the hourly schedule and the AI briefing (offline)."""
import json
import time

import pandas as pd
import pytest

from solarcast.nz import config, grid as nzgrid, model as nzmodel, service as svc
from solarcast.nz.features import build_features
from conftest import make_long

CELLS = [{"lat": -36.75, "lon": 174.75, "qlat": -36.8, "qlon": 174.7},     # Auckland
         {"lat": -41.25, "lon": 175.25},                                     # Wairarapa
         {"lat": -43.75, "lon": 172.25},                                     # Canterbury
         {"lat": -45.25, "lon": 168.75}]                                     # Queenstown


@pytest.fixture(scope="module")
def service(long_df):
    feat = build_features(long_df)
    m = nzmodel.train(feat[feat.index.get_level_values("target_time") < "2025-12-01"], max_iter=40)
    return svc.Service(m, long_df, meta={})


@pytest.fixture()
def fake_fetch(monkeypatch):
    calls = []

    def fetch_points(points, names, start=None, end=None, **kw):
        calls.append((len(points), start, end))
        if start:
            g = make_long(start, (pd.Timestamp(end) + pd.Timedelta("1D")).strftime("%Y-%m-%d"), ["Wellington"], seed=4)
        else:
            now = pd.Timestamp.now(tz="UTC").floor("h")
            g = make_long((now - pd.Timedelta("3D")).strftime("%Y-%m-%d"),
                          (now + pd.Timedelta("2D")).strftime("%Y-%m-%d"), ["Wellington"], seed=5)
        return pd.concat([g.assign(region=n) for n in names], ignore_index=True)

    monkeypatch.setattr(nzgrid.data, "fetch_points", fetch_points)
    return calls


def test_island_split():
    assert nzgrid.island(-36.8, 174.7) == "North" and nzgrid.island(-41.29, 174.78) == "North"     # Auckland, Wellington
    assert nzgrid.island(-41.5, 173.96) == "South" and nzgrid.island(-43.5, 172.6) == "South"      # Blenheim, Christchurch
    assert nzgrid.island(-46.9, 168.1) == "South"                                                  # Stewart Island


def test_national_stats_and_template():
    cells = [{"lat": -36.8, "lon": 174.8, "near": "Auckland", "island": "North", "kw": 10.0, "cloud": 80, "actual": None},
             {"lat": -43.5, "lon": 172.5, "near": "Christchurch", "island": "South", "kw": 40.0, "cloud": 10, "actual": None},
             {"lat": -45.0, "lon": 168.7, "near": "Queenstown", "island": "South", "kw": 0.5, "cloud": 100, "actual": None}]
    s = nzgrid.national_stats(cells)
    assert s["cells"] == 3 and s["best_area"]["near"] == "Christchurch" and s["max_kw"] == 40.0
    assert s["island_mean_kw"] == {"North": 10.0, "South": 20.2} and s["share_generating_pct"] == 67
    assert s["areas_best"][0]["around"] == "Christchurch" and "mean_abs_error_kw" not in s
    text = nzgrid.template_summary(s, "2026-01-10 12:00")
    assert "Christchurch" in text and "North Island 10.0" in text
    night = nzgrid.national_stats([dict(c, kw=0.0) for c in cells])
    assert "no part of New Zealand" in nzgrid.template_summary(night, "2026-01-10 23:00")


def test_shipped_cells_cover_new_zealand():
    step, cells = nzgrid.load_cells()
    assert step == 0.5 and 80 <= len(cells) <= 200
    assert all(svc.NZ_BOX[0] <= c["lat"] <= svc.NZ_BOX[1] and svc.NZ_BOX[2] <= c["lon"] <= svc.NZ_BOX[3] for c in cells)
    isl = {nzgrid.island(c["lat"], c["lon"]) for c in cells}
    assert isl == {"North", "South"}


def test_replay_snapshot_is_cached_and_budgeted(service, fake_fetch, monkeypatch):
    g = nzgrid.NationalGrid(service, CELLS, 0.5)
    t = service.default_time()
    snap = g.snapshot(t)
    assert len(snap["cells"]) == 4 and snap["time"] == t and not snap["live"] and snap["step"] == 0.5
    assert {c["near"] for c in snap["cells"]} >= {"Auckland", "Christchurch"}
    assert all(c["actual"] is not None for c in snap["cells"]) and "mean_abs_error_kw" in snap["stats"]
    assert snap["briefing"]["mode"] == "template" and snap["briefing"]["text"]
    json.dumps(snap)
    later = (pd.Timestamp(t) + pd.Timedelta("1h")).strftime(svc.ISO)
    g.snapshot(later)                                         # same local day: no new download
    assert len(fake_fetch) == 1 and fake_fetch[0][0] == 4
    monkeypatch.setattr(nzgrid, "MAX_NEW_DAYS_PER_HOUR", 1)
    with pytest.raises(RuntimeError):
        g.snapshot((pd.Timestamp(t) + pd.Timedelta("5D")).strftime(svc.ISO))


def test_live_snapshot_and_precompute(service, fake_fetch):
    g = nzgrid.NationalGrid(service, CELLS, 0.5)
    done = g.precompute(service.default_time(), live_in_background=False)
    assert done["live"] == "ok" and done["replay_day"] == "ok" and g.status["state"] == "ready"
    snap = g.snapshot("live")
    assert snap["live"] and snap["time"] == pd.Timestamp.now(tz="UTC").floor("h").strftime(svc.ISO)
    assert all(c["actual"] is None for c in snap["cells"])
    nxt = (pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta("1h")).strftime(svc.ISO)
    assert g.snapshot("live", at=nxt)["time"] == nxt        # precomputed at :50 for the coming hour
    assert len(fake_fetch) == 2


def test_next_refresh_is_at_minute_50():
    t = pd.Timestamp("2026-01-10 03:20", tz="UTC")
    assert nzgrid.NationalGrid.next_refresh(t) == pd.Timestamp("2026-01-10 03:50", tz="UTC")
    assert nzgrid.NationalGrid.next_refresh(pd.Timestamp("2026-01-10 03:55", tz="UTC")) == pd.Timestamp("2026-01-10 04:50", tz="UTC")


def test_ai_briefing_is_written_in_the_background(service, fake_fetch):
    from solarcast.nz.assistant import Assistant

    class LLM:
        name = "fake"

        def __init__(self):
            self.calls = []

        def chat(self, messages):
            self.calls.append(messages)
            return "Sunniest around Christchurch."

    llm = LLM()
    g = nzgrid.NationalGrid(service, CELLS, 0.5)
    a = Assistant(service, llm, grid=g)
    g.summarizer = a.summarize
    t = service.default_time()
    first = g.snapshot(t)
    assert first["briefing"]["mode"] == "template" and first["briefing_pending"]
    for _ in range(50):
        if g.snapshot(t, summarize=False)["briefing"]["mode"] == "llm":
            break
        time.sleep(0.05)
    snap = g.snapshot(t)
    assert snap["briefing"] == {"text": "Sunniest around Christchurch.", "mode": "llm", "model": "fake"}
    facts = json.loads(llm.calls[0][1]["content"].split("\n", 1)[1])
    assert facts["best_area"]["near"] and facts["workshop_load_kw"] == config.WORKSHOP_LOAD_KW
    assert "nationwide_grid" in a.facts(t, None)


def test_grid_route(service, fake_fetch):
    pytest.importorskip("flask")
    from solarcast.nz.webapp import create_app

    g = nzgrid.NationalGrid(service, CELLS, 0.5)
    client = create_app(service, grid=g).test_client()
    r = client.get("/api/grid?t=" + service.default_time())
    assert r.status_code == 200 and len(r.get_json()["cells"]) == 4
    assert create_app(service).test_client().get("/api/grid").status_code == 404


def test_live_grid_in_background_and_shipped_replay(service, fake_fetch, monkeypatch, tmp_path):
    """The dashboard link must not wait for downloads: replay uses shipped data, live is computed in a thread."""
    t = service.default_time()
    day = pd.Timestamp(t).tz_convert(config.LOCAL_TZ).normalize()
    names = [f"cell{i:03d}" for i in range(len(CELLS))]
    shipped = nzgrid.data.fetch_points(CELLS, names, (day - pd.Timedelta("3D")).strftime("%Y-%m-%d"),
                                       (day + pd.Timedelta("3D")).strftime("%Y-%m-%d"))
    path = tmp_path / "grid_replay.csv.gz"
    shipped.to_csv(path, index=False)
    monkeypatch.setattr(nzgrid, "GRID_REPLAY_PATH", path)
    fake_fetch.clear()
    g = nzgrid.NationalGrid(service, CELLS, 0.5)
    done = g.precompute(t)
    assert done["replay_day"] == "ok" and all(c[1] is None for c in fake_fetch)   # replay: no download at all
    for _ in range(100):
        if not g._live_busy.is_set():
            break
        time.sleep(0.05)
    assert g.snapshot("live")["live"] and len(fake_fetch) == 1


def test_live_grid_still_computing_is_a_clear_error(service, fake_fetch):
    g = nzgrid.NationalGrid(service, CELLS, 0.5)
    g._live_busy.set()
    with pytest.raises(RuntimeError, match="still being computed"):
        g.snapshot("live")
