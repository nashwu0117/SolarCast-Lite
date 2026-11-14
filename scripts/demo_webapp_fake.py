"""Run the SolarCast NZ web app with a FakeService (synthetic data) to test the UI without a model.

Usage:  PYTHONPATH=src python scripts/demo_webapp_fake.py [--port 8000] [--no-tunnel]
The numbers are made up (a smooth sun curve plus pseudo-random cloud); they only exercise the interface.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NZ = ZoneInfo("Pacific/Auckland")
REGIONS = {
    "Whangarei": (-35.73, 174.32), "Auckland": (-36.85, 174.76), "Hamilton": (-37.79, 175.28),
    "Tauranga": (-37.69, 176.17), "Gisborne": (-38.66, 178.02), "Napier": (-39.49, 176.91),
    "New Plymouth": (-39.06, 174.08), "Palmerston North": (-40.36, 175.61), "Wellington": (-41.29, 174.78),
    "Nelson": (-41.27, 173.28), "Christchurch": (-43.53, 172.64), "Queenstown": (-45.03, 168.66),
    "Dunedin": (-45.87, 170.50), "Invercargill": (-46.41, 168.35),
}
SYSTEM_KW = 50.0
MODELS = ["Our model", "Persistence", "Clear-sky persistence"]


def _rand(*key) -> float:
    """Deterministic pseudo-random number in [0, 1) from a key."""
    return int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16) / 16 ** 8


def _parse(t: str) -> datetime:
    if t == "live":
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone.utc)


def _weather(region: str, when_utc: datetime) -> dict:
    """Synthetic weather and power at one hour."""
    lat, _ = REGIONS[region]
    local = when_utc.astimezone(NZ)
    hour = local.hour + local.minute / 60
    sun = max(0.0, math.sin(math.pi * (hour - 6) / 14)) if 6 < hour < 20 else 0.0
    sun *= 0.75 + 0.25 * math.cos(math.radians(abs(lat) - 35))  # a little lower in the south
    cloud = 100 * (0.5 + 0.5 * math.sin(local.toordinal() * 1.7 + len(region)))
    cloud = min(100, max(0, cloud * 0.8 + 40 * _rand(region, when_utc.isoformat()) - 10))
    ghi = 1000 * sun * (1 - 0.75 * cloud / 100)
    return {"local": local, "ghi": ghi, "cloud": cloud, "is_day": sun > 0.02,
            "temp": 8 + 10 * sun + 8 * (1 - abs(lat) / 50), "wind": 10 + 20 * _rand(region, "w", when_utc.date().isoformat()),
            "kw": min(SYSTEM_KW, SYSTEM_KW * ghi / 1000 * 0.8)}


class FakeService:
    """Same interface as solarcast.nz.service.Service, with synthetic data."""

    def __init__(self):
        self.start = datetime(2026, 1, 12, 0, tzinfo=timezone.utc)
        self._times = [(self.start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ") for h in range(24 * 7)]

    def available_times(self):
        return self._times

    def default_time(self):
        return "2026-01-15T23:00:00Z"  # 12:00 NZ local time on 16 January

    def snapshot(self, time_utc):
        now = _parse(time_utc)
        live = time_utc == "live"
        rows = []
        for name, (lat, lon) in REGIONS.items():
            w, nxt, prev = _weather(name, now), _weather(name, now + timedelta(hours=1)), _weather(name, now)
            predicted = max(0.0, nxt["kw"] * (0.9 + 0.2 * _rand(name, "p", now.isoformat())))
            rows.append({
                "name": name, "lat": lat, "lon": lon, "local_time": w["local"].strftime("%a %d %b %H:%M"),
                "cloud_cover_pct": w["cloud"], "temp_c": w["temp"], "wind_kmh": w["wind"], "ghi_now_wm2": w["ghi"],
                "is_day": w["is_day"], "predicted_kw_next": predicted, "persistence_kw_next": prev["kw"],
                "actual_kw_next": None if live else nxt["kw"], "generating": predicted > 1,
            })
        return rows

    def detail(self, region, time_utc):
        if region not in REGIONS:
            raise KeyError(region)
        now = _parse(time_utc)
        midnight = now.astimezone(NZ).replace(hour=0, minute=0, second=0, microsecond=0)
        hours, actual, pred, pers, cloud = [], [], [], [], []
        for h in range(24):
            t = (midnight + timedelta(hours=h)).astimezone(timezone.utc)
            w, prev = _weather(region, t), _weather(region, t - timedelta(hours=1))
            hours.append(h)
            actual.append(None if (time_utc == "live" and t > now) else w["kw"])
            pred.append(max(0.0, w["kw"] * (0.9 + 0.2 * _rand(region, "p", t.isoformat()))))
            pers.append(prev["kw"])
            cloud.append(w["cloud"])
        nxt = _weather(region, now + timedelta(hours=1))
        charge = nxt["kw"] > 11
        reason = (f"Predicted solar {nxt['kw']:.1f} kW covers the 11 kW charger, so charge now."
                  if charge else f"Predicted solar only {nxt['kw']:.1f} kW; waiting for a sunnier hour saves grid power.")
        kwh = sum(x for x in pred[8:18]) * 0.5
        return {"region": region, "hours": hours, "actual_kw": actual, "predicted_kw": pred, "persistence_kw": pers,
                "predicted_kw_next": nxt["kw"],
                "cloud_cover_pct": cloud, "charge_now": charge, "reason": reason,
                "grid_kwh_saved": kwh, "money_saved_nzd": kwh * 0.30, "co2_saved_kg": kwh * 0.10,
                "assumptions": "price NZD 0.30/kWh, grid 0.10 kg CO2/kWh, EV 40 kWh/day, 11 kW charger, 08:00-18:00 local",
                "metrics": {"mae_kw": 1.8 + _rand(region, "m"), "rmse_kw": 3.0 + _rand(region, "r")}}

    def model_report(self):
        factor = {"Our model": 1.0, "Persistence": 1.7, "Clear-sky persistence": 1.35}
        regions = list(REGIONS)
        sample_start = datetime(2026, 1, 12, tzinfo=timezone.utc)
        times = [(sample_start + timedelta(hours=h)) for h in range(24 * 7)]
        region = "Wellington"
        return {
            "periods": {"train": ["2022-01-01", "2025-01-01"], "val": ["2025-01-01", "2025-07-01"], "test": ["2025-07-01", "2026-11-01"]},
            "overall": [{"model": m, "mae_kw": 2.1 * f, "mae_wm2": 52 * f, "rmse_kw": 3.9 * f} for m, f in factor.items()],
            "per_region": [{"region": r, "model": m, "mae_kw": (1.6 + 1.2 * _rand(r, "e")) * f}
                           for r in regions for m, f in factor.items()],
            "importance": [{"feature": f, "importance": v} for f, v in [
                ("clear_sky_ghi_target", 0.31), ("ghi_t", 0.22), ("cloud_cover_t", 0.12), ("ghi_lag1", 0.09), ("sun_elevation_target", 0.08),
                ("ghi_roll3", 0.06), ("temp_t", 0.04), ("humidity_t", 0.03), ("hour_sin", 0.03), ("latitude", 0.02)]],
            "sample": {"region": region, "times": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times],
                       "actual_kw": [_weather(region, t + timedelta(hours=1))["kw"] for t in times],
                       "predicted_kw": [_weather(region, t + timedelta(hours=1))["kw"] * (0.9 + 0.2 * _rand("s", t.isoformat())) for t in times],
                       "persistence_kw": [_weather(region, t)["kw"] for t in times]},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-tunnel", action="store_true")
    args = parser.parse_args()
    from solarcast.nz.webapp import serve

    url = serve(FakeService(), port=args.port, tunnel=not args.no_tunnel)
    print(f"URL: {url}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
