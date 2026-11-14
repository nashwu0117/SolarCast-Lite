"""Web app routes with a FakeService (no model, no network)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("flask")
from solarcast.nz.webapp import create_app  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "demo_webapp_fake.py"
_spec = importlib.util.spec_from_file_location("demo_webapp_fake", _SCRIPT)
_fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fake)


@pytest.fixture()
def client():
    return create_app(_fake.FakeService()).test_client()


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "leaflet" in html and "chart.js" in html
    assert "Prototype; not energy or financial advice." in html


def test_times(client):
    data = client.get("/api/times").get_json()
    assert data["default"] in data["times"]


def test_snapshot_replay_and_live(client):
    default = client.get("/api/times").get_json()["default"]
    for t in (default, "live"):
        res = client.get(f"/api/snapshot?t={t}")
        assert res.status_code == 200
        regions = res.get_json()["regions"]
        assert len(regions) == 14
        for key in ("name", "lat", "lon", "local_time", "cloud_cover_pct", "temp_c", "wind_kmh", "ghi_now_wm2",
                    "is_day", "predicted_kw_next", "persistence_kw_next", "actual_kw_next", "generating"):
            assert key in regions[0]
    # live mode has no actual value
    assert regions[0]["actual_kw_next"] is None


def test_detail(client):
    res = client.get("/api/detail?region=Wellington&t=" + client.get("/api/times").get_json()["default"])
    assert res.status_code == 200
    d = res.get_json()
    for key in ("hours", "actual_kw", "predicted_kw", "persistence_kw", "cloud_cover_pct", "charge_now", "reason",
                "grid_kwh_saved", "money_saved_nzd", "co2_saved_kg"):
        assert key in d
    assert len(d["hours"]) == len(d["predicted_kw"])


def test_detail_errors(client):
    assert client.get("/api/detail?region=Atlantis").status_code == 404
    assert client.get("/api/detail").status_code == 400


def test_model_report(client):
    d = client.get("/api/model").get_json()
    for key in ("periods", "overall", "per_region", "importance", "sample"):
        assert key in d
    assert {"model", "mae_kw", "mae_wm2", "rmse_kw"} <= set(d["overall"][0])
    assert {"region", "model", "mae_kw"} <= set(d["per_region"][0])
    assert {"times", "actual_kw", "predicted_kw", "persistence_kw", "region"} <= set(d["sample"])


def test_nan_becomes_null(client):
    class NanService(_fake.FakeService):
        def snapshot(self, t):
            rows = super().snapshot(t)
            rows[0]["temp_c"] = float("nan")
            return rows

    res = create_app(NanService()).test_client().get("/api/snapshot?t=live")
    assert res.get_json()["regions"][0]["temp_c"] is None
