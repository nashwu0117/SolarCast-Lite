"""Nationwide grid: next-hour forecasts for every 0.5-degree land cell of New Zealand, computed ahead of time.

Why: the 14 region circles only show 14 places. The grid covers the whole country (about 120 cells, roughly 50 km
each), so the map opens as a full heat map instead of 14 dots.

When it is computed:
- At start-up (`precompute`): the live hours and the default replay day, so the page is ready the moment it opens.
- Every hour at :50 (`start_hourly`): the live data is downloaded again and every cell is recomputed, so the next
  hour's map is ready before that hour starts. The page polls and switches by itself.
- Other replay days are computed on demand (one download per day, cached) with an hourly budget, because every new
  day costs about two Open-Meteo calls per cell and the free API allows about 10,000 calls per day.

Each cell is run through the SAME model as the regions. A cell borrows its nearest region's id (its climate) and
uses its own latitude and sun position, exactly like a clicked map point. Cells were chosen once with Open-Meteo's
elevation service (scripts/build_grid_cells.py): a cell is land when at least 2 of 5 sample points are above sea level.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from solarcast.nz import config, data, model as nzmodel
from solarcast.nz.features import build_features
from solarcast.nz.service import HOUR, ISO, nearest_region

GRID_PATH = config.ROOT / "data" / "nz" / "grid_cells.json"
GRID_REPLAY_PATH = config.ROOT / "data" / "nz" / "grid_replay.csv.gz"   # cell weather around the default replay day
REFRESH_MINUTE = 50                 # live data is downloaded again at HH:50, ready for the next hour
MAX_DAYS_CACHED = 12                # replay days kept in memory
MAX_NEW_DAYS_PER_HOUR = 6           # replay days downloaded on demand per hour (protects the free API quota)


def load_cells(path=GRID_PATH) -> tuple[float, list[dict]]:
    """(step in degrees, cells). Each cell: lat/lon = cell centre (drawn), qlat/qlon = land point the weather is for."""
    spec = json.loads(path.read_text())
    return float(spec["step"]), spec["cells"]


def island(lat: float, lon: float) -> str:
    """North or South Island (Cook Strait split; Stewart Island counts as South)."""
    return "South" if lat < -40.4 and (lon < 174.4 or lat < -41.6) else "North"


def national_stats(cells: list[dict]) -> dict:
    """Numbers that summarise the whole country for one hour (they feed the briefing and the chat)."""
    if not cells:
        return {}
    kw = np.array([c["kw"] for c in cells], float)
    best = cells[int(np.argmax(kw))]
    out = {"cells": len(cells), "mean_kw": round(float(kw.mean()), 1), "max_kw": round(float(kw.max()), 1),
           "best_area": {"near": best["near"], "lat": best["lat"], "lon": best["lon"], "kw": best["kw"]},
           "share_generating_pct": round(float((kw > 1).mean() * 100)),
           "share_above_workshop_load_pct": round(float((kw > config.WORKSHOP_LOAD_KW).mean() * 100))}
    clouds = [c["cloud"] for c in cells if c.get("cloud") is not None]
    out["mean_cloud_pct"] = round(float(np.mean(clouds))) if clouds else None
    out["island_mean_kw"] = {k: round(float(np.mean([c["kw"] for c in cells if c["island"] == k])), 1)
                             for k in ("North", "South") if any(c["island"] == k for c in cells)}
    by = {}
    for c in cells:
        by.setdefault(c["near"], []).append(c["kw"])
    areas = sorted(((n, round(float(np.mean(v)), 1)) for n, v in by.items()), key=lambda x: -x[1])
    out["areas_best"] = [{"around": n, "mean_kw": v} for n, v in areas[:3]]
    out["areas_worst"] = [{"around": n, "mean_kw": v} for n, v in areas[-3:][::-1]]
    actual = [c["actual"] for c in cells if c.get("actual") is not None]
    if len(actual) == len(cells):
        out["mean_actual_kw"] = round(float(np.mean(actual)), 1)
        out["mean_abs_error_kw"] = round(float(np.mean([abs(c["kw"] - c["actual"]) for c in cells])), 2)
    return out


def template_summary(stats: dict, local_time: str) -> str:
    """A fixed-wording briefing built from the numbers (shown until, or instead of, the AI briefing)."""
    if not stats:
        return "No grid data for this hour."
    if stats["max_kw"] <= 1:
        return (f"For the hour after {local_time} (NZ time) no part of New Zealand is generating solar power "
                f"(night or heavy cloud); every rooftop system would give about zero.")
    isl = stats.get("island_mean_kw", {})
    b = stats["best_area"]
    pct = lambda kw: kw / config.SYSTEM_KW * 100
    parts = [f"For the hour after {local_time} (NZ time) the sunniest area is around {b['near']}: solar panels there give "
             f"about {pct(b['kw']):.0f}% of their rated power (a 5 kW home system about {b['kw'] / 10:.1f} kW, a 50 kW "
             f"business system about {b['kw']:.1f} kW).",
             f"New Zealand average: {pct(stats['mean_kw']):.0f}% of rated power"
             + (f" (North Island {pct(isl['North']):.0f}%, South Island {pct(isl['South']):.0f}%)." if len(isl) == 2 else "."),
             f"Panels are generating in {stats['share_generating_pct']}% of the country."]
    return " ".join(parts)


def _percent(stats: dict) -> dict:
    """Best / mean / per-island output as a percent of rated power: the same for a 5 kW home or a 50 kW business."""
    if not stats:
        return {}
    pct = lambda kw: round(kw / config.SYSTEM_KW * 100)
    return {"best_area": pct(stats["best_area"]["kw"]), "new_zealand_mean": pct(stats["mean_kw"]),
            **{f"{k.lower()}_island_mean": pct(v) for k, v in stats.get("island_mean_kw", {}).items()}}


def _scaled(stats: dict, size_kw: float) -> dict:
    """The same numbers for a system of size_kw (output scales linearly with panel size)."""
    if not stats:
        return {}
    f = size_kw / config.SYSTEM_KW
    return {"best_area_kw": round(stats["best_area"]["kw"] * f, 2), "new_zealand_mean_kw": round(stats["mean_kw"] * f, 2)}


class NationalGrid:
    """Forecasts for all land cells, for live hours and replay days. Thread-safe; computed ahead of time."""

    def __init__(self, service, cells: list[dict] | None = None, step: float | None = None):
        if cells is None:
            step, cells = load_cells()
        self.service, self.step, self.cells = service, float(step or 0.5), cells
        self.names = [f"cell{i:03d}" for i in range(len(cells))]
        self._info = {}
        for name, c in zip(self.names, cells):
            near, _ = nearest_region(c.get("qlat", c["lat"]), c.get("qlon", c["lon"]))
            self._info[name] = {"lat": c["lat"], "lon": c["lon"], "near": near, "island": island(c["lat"], c["lon"]),
                                "region_id": list(config.REGIONS).index(near)}
        self._live = None                                      # {"frame", "at"}
        self._days: OrderedDict = OrderedDict()                # local date -> frame
        self._new_days: list[float] = []
        self._fetch_lock = threading.Lock()
        self._summaries: dict = {}                             # (time iso) -> {"text", "mode", "model"}
        self._pending: set = set()
        self._sum_lock = threading.Lock()
        self.summarizer = None                                 # callable(facts) -> {"text", "mode", "model"} | None
        self.status = {"state": "idle", "updated_at": None, "next_update": None, "error": None}
        self._thread = None
        self._replay_raw = None                                # shipped cell weather (no download needed)
        self._live_busy = threading.Event()

    # ----- computing -----
    def _compute(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Long weather frame for the cells -> per (cell, target_time): predicted kW, cloud, actual kW."""
        info = pd.DataFrame.from_dict(self._info, orient="index")
        raw = raw.assign(loc_lat=raw["region"].map(lambda n: self.cells[int(n[4:])].get("qlat", info.at[n, "lat"])),
                         loc_lon=raw["region"].map(lambda n: self.cells[int(n[4:])].get("qlon", info.at[n, "lon"])),
                         loc_region_id=raw["region"].map(info["region_id"]))
        feat = build_features(raw, require_target=False)
        pred = nzmodel.predict(self.service.model, feat)
        out = pd.DataFrame({"kw": nzmodel.to_kw(pred, feat["temp_t"]), "cloud": feat["cloud_t"],
                            "actual": feat["target_kw"]}, index=feat.index)
        return out

    def refresh_live(self) -> None:
        raw = data.fetch_points([(c.get("qlat", c["lat"]), c.get("qlon", c["lon"])) for c in self.cells], self.names)
        frame = self._compute(raw)
        now = pd.Timestamp.now(tz="UTC").floor("h")
        frame.loc[frame.index.get_level_values(1) > now, "actual"] = np.nan    # hours that have not happened yet
        self._live = {"frame": frame, "at": time.time(), "raw": raw, "ahead": {}}

    def _ahead_frame(self, h: int) -> pd.DataFrame:
        """Live cells for the hour ending h hours after each issue hour (the service's h-hours-ahead model)."""
        cache = self._live.setdefault("ahead", {})
        if h not in cache:
            info = pd.DataFrame.from_dict(self._info, orient="index")
            raw = self._live["raw"]
            raw = raw.assign(loc_lat=raw["region"].map(lambda n: self.cells[int(n[4:])].get("qlat", info.at[n, "lat"])),
                             loc_lon=raw["region"].map(lambda n: self.cells[int(n[4:])].get("qlon", info.at[n, "lon"])),
                             loc_region_id=raw["region"].map(info["region_id"]))
            feat = build_features(raw, require_target=False, horizon=h)
            pred = nzmodel.predict(self.service.ahead_models[h], feat)
            cache[h] = pd.DataFrame({"kw": nzmodel.to_kw(pred, feat["temp_t"]), "cloud": feat["cloud_t"],
                                     "actual": np.nan}, index=feat.index)
        return cache[h]

    def _day_frame(self, t: pd.Timestamp, budget: bool = True) -> pd.DataFrame:
        day = t.tz_convert(config.LOCAL_TZ).normalize()
        key = str(day.date())
        if key in self._days:
            self._days.move_to_end(key)
            return self._days[key]
        with self._fetch_lock:
            if key in self._days:
                return self._days[key]
            now = time.time()
            self._new_days = [x for x in self._new_days if now - x < 3600]
            if budget and len(self._new_days) >= MAX_NEW_DAYS_PER_HOUR:
                raise RuntimeError("The nationwide grid has downloaded several new days this hour; to protect the free "
                                   "weather API it will compute more later. The region circles still work.")
            start = (day - pd.Timedelta("2D")).tz_convert("UTC").strftime("%Y-%m-%d")
            end = (day + pd.Timedelta("2D")).tz_convert("UTC").strftime("%Y-%m-%d")
            raw = self._shipped(start, end)
            if raw is None:
                pts = [(c.get("qlat", c["lat"]), c.get("qlon", c["lon"])) for c in self.cells]
                raw = data.fetch_points(pts, self.names, start, end)
                self._new_days.append(now)
            frame = self._compute(raw)
            self._days[key] = frame
            while len(self._days) > MAX_DAYS_CACHED:
                self._days.popitem(last=False)
            return frame

    def _shipped(self, start: str, end: str):
        """Cell weather shipped in the repository for [start, end] (dates), or None if it does not cover them."""
        if self._replay_raw is None:
            try:
                df = pd.read_csv(GRID_REPLAY_PATH)
                df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
                if set(df["region"]) != set(self.names):           # a different cell list: do not use it
                    df = df.iloc[0:0]
                self._replay_raw = df
            except (OSError, ValueError, KeyError):
                self._replay_raw = pd.DataFrame()
        df = self._replay_raw
        if df.empty:
            return None
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta("23h")
        if df["time_utc"].min() > lo or df["time_utc"].max() < hi:
            return None
        return df[(df["time_utc"] >= lo) & (df["time_utc"] <= hi)].reset_index(drop=True)

    def _refresh_live_safe(self) -> None:
        self._live_busy.set()
        try:
            self.refresh_live()
            self.status.update(state="ready", updated_at=pd.Timestamp.now(tz="UTC").strftime(ISO), error=None)
        except Exception as exc:                               # replay still works without the internet
            self.status.update(state="error", error=f"live grid download failed ({type(exc).__name__})")
        finally:
            self._live_busy.clear()

    def precompute(self, replay_time: str | None = None, live_in_background: bool = True) -> dict:
        """Compute the default replay day now (shipped data, a few seconds) and the live hours (in the background
        by default, so the dashboard link appears at once; the live map shows up when it is ready)."""
        t0, done = time.time(), {}
        if live_in_background:
            self._live_busy.set()
            threading.Thread(target=self._refresh_live_safe, name="solarcast-grid-live", daemon=True).start()
            done["live"] = "computing in the background"
        else:
            self._refresh_live_safe()
            done["live"] = "ok" if self._live is not None else self.status.get("error")
        recent = replay_time and self._issue(replay_time) > pd.Timestamp.now(tz="UTC") - pd.Timedelta("47h")
        if recent:
            done["replay_day"] = "the past 24 hours come with the live download"
        elif replay_time:
            try:
                self._day_frame(self._issue(replay_time), budget=False)
                done["replay_day"] = "ok"
            except Exception as exc:
                done["replay_day"] = f"failed ({type(exc).__name__})"
        done["seconds"] = round(time.time() - t0, 1)
        done["cells"] = len(self.cells)
        return done

    # ----- hourly refresh -----
    @staticmethod
    def next_refresh(now: pd.Timestamp | None = None) -> pd.Timestamp:
        now = now or pd.Timestamp.now(tz="UTC")
        nxt = now.floor("h") + pd.Timedelta(minutes=REFRESH_MINUTE)
        return nxt if nxt > now else nxt + HOUR

    def start_hourly(self) -> None:
        """Background thread: at every HH:50 download the live data again and recompute every cell."""
        if self._thread and self._thread.is_alive():
            return

        def loop():
            while True:
                nxt = self.next_refresh()
                self.status["next_update"] = nxt.strftime(ISO)
                time.sleep(max(1.0, (nxt - pd.Timestamp.now(tz="UTC")).total_seconds()))
                try:
                    self.refresh_live()
                    self.status.update(state="ready", updated_at=pd.Timestamp.now(tz="UTC").strftime(ISO), error=None)
                    nxt_hour = (pd.Timestamp.now(tz="UTC").floor("h") + HOUR).strftime(ISO)
                    self._summarize(self.snapshot("live", at=nxt_hour, summarize=False))   # briefing ready at :00
                except Exception as exc:
                    self.status.update(state="error", error=f"live grid refresh failed ({type(exc).__name__})")

        self.status["next_update"] = self.next_refresh().strftime(ISO)
        self._thread = threading.Thread(target=loop, name="solarcast-grid", daemon=True)
        self._thread.start()

    # ----- reading -----
    @staticmethod
    def _issue(time_utc) -> pd.Timestamp:
        t = pd.Timestamp(time_utc)
        return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).floor("h")

    def wait_live(self, timeout_s: float = 600) -> bool:
        """Block until the live grid computed in the background is ready (True) or timeout_s passes (False)."""
        end = time.time() + timeout_s
        while self._live_busy.is_set() and time.time() < end:
            time.sleep(1)
        return self._live is not None

    def snapshot(self, time_utc, at: str | None = None, summarize: bool = True) -> dict:
        """All cells for one issue hour t (forecast for [t, t+1h)), with national numbers and the briefing."""
        ahead = int(str(time_utc).split(":", 1)[1]) if str(time_utc).startswith("ahead:") else 1
        live = str(time_utc).lower() == "live" or ahead > 1 or str(time_utc) == "ahead:1"
        recent = False
        if not live and self._live is not None:
            t0 = self._issue(time_utc)
            recent = (t0 + HOUR) in set(self._live["frame"].index.get_level_values(1)) and t0 < pd.Timestamp.now(tz="UTC").floor("h")
        if live:
            if self._live is None:
                if self._live_busy.is_set():
                    raise RuntimeError("The live nationwide grid is still being computed (about a minute). "
                                       "The region circles and replay already work.")
                with self._fetch_lock:
                    if self._live is None:
                        self._refresh_live_safe()
                if self._live is None:
                    raise RuntimeError(self.status.get("error") or "The live nationwide grid is not available.")
            frame = self._live["frame"]
            t = self._issue(at) if at else pd.Timestamp.now(tz="UTC").floor("h")
            times = frame.index.get_level_values(1)
            if t + HOUR not in set(times):                     # e.g. data older than expected: newest available
                t = times.max() - HOUR
            if ahead > 1 and ahead in getattr(self.service, "ahead_models", {}):
                frame = self._ahead_frame(ahead)
        elif recent:                                           # one of the past 24 hours: the live download has it
            t, frame = self._issue(time_utc), self._live["frame"]
        else:
            t = self._issue(time_utc)
            frame = self._day_frame(t)
        tt = t + (ahead if live else 1) * HOUR
        rows = frame.xs(tt, level=1, drop_level=True) if tt in set(frame.index.get_level_values(1)) else None
        cells = []
        if rows is not None:
            for name, r in rows.iterrows():
                i = self._info[name]
                cells.append({"lat": i["lat"], "lon": i["lon"], "near": i["near"], "island": i["island"],
                              "kw": round(float(r["kw"]), 1),
                              "cloud": None if pd.isna(r["cloud"]) else int(round(r["cloud"])),
                              "actual": None if pd.isna(r["actual"]) else round(float(r["actual"]), 1)})
        local = t.tz_convert(config.LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        stats = national_stats(cells)
        key = ("live|" if live else "replay|") + t.strftime(ISO) + (f"|ahead{ahead}" if ahead > 1 else "")
        res = {"time": t.strftime(ISO), "local_time": local, "live": live, "step": self.step, "cells": cells,
               "stats": stats, "briefing": {"text": template_summary(stats, local), "mode": "template"},
               "updated_at": self.status.get("updated_at"), "next_update": self.status.get("next_update"),
               "refresh_minute": REFRESH_MINUTE, "key": key}
        ai = self._summaries.get(key)
        if ai:
            res["briefing"] = ai
        elif summarize and stats and self.summarizer is not None:
            res["briefing_pending"] = self._summarize_async(res)
        return res

    # ----- AI briefing -----
    def facts(self, snap: dict) -> dict:
        return {"nz_local_time": snap["local_time"], "mode": "live" if snap["live"] else "replay of a past hour",
                "forecast_for": "the hour after this time", "system_kw": config.SYSTEM_KW,
                "kw_values_are_for": f"a {config.SYSTEM_KW:g} kW reference system; percent_of_rated_power works for any size",
                "percent_of_rated_power": _percent(snap["stats"]),
                "home_5kw_system_kw": _scaled(snap["stats"], 5.0),
                "workshop_load_kw": config.WORKSHOP_LOAD_KW, "grid_cell_size_deg": snap["step"], **snap["stats"]}

    def _summarize(self, snap: dict) -> None:
        if self.summarizer is None or not snap.get("stats"):
            return
        out = self.summarizer(self.facts(snap))
        if out and out.get("mode") == "llm":
            with self._sum_lock:
                if len(self._summaries) > 300:
                    self._summaries.clear()
                self._summaries[snap["key"]] = out

    def _summarize_async(self, snap: dict) -> bool:
        """Start writing the AI briefing in the background; True while one is being written for this hour."""
        ready = getattr(self.summarizer, "__self__", None)
        if ready is not None and hasattr(ready, "llm_ready") and not ready.llm_ready():
            return False
        with self._sum_lock:
            if snap["key"] in self._pending:
                return True
            if self._pending:                                  # one briefing at a time
                return False
            self._pending.add(snap["key"])

        def run():
            try:
                self._summarize(snap)
            finally:
                with self._sum_lock:
                    self._pending.discard(snap["key"])

        threading.Thread(target=run, name="solarcast-briefing", daemon=True).start()
        return True
