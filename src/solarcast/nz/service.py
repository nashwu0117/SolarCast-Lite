"""What the web app calls: replay any past hour of the history, or go live, for all regions.

`time_utc` is the ISSUE hour t (UTC ISO string, e.g. "2026-01-15T00:00:00Z"): we know everything up to the end of
hour t and forecast the hour [t, t+1h). "live" means the latest real hour from the Open-Meteo forecast API.
"ahead:h" (h = 1..3) is "Predict next": from the latest real hour, the hour that ends h hours later.

With recent_replay=True (the dashboard) Replay covers the PAST 24 HOURS of live data, with what really happened; if the
live download fails it falls back to the replay history file.
"""
import threading
import time

import numpy as np
import pandas as pd

from solarcast.nz import config, data, model as nzmodel
from solarcast.nz.features import FEATURES, build_features, daytime_mask

HOUR = pd.Timedelta("1h")
ISO = "%Y-%m-%dT%H:%M:%SZ"
LIVE_TTL_S = 600
FIXED_WINDOW = (10, 14)                      # comparison habit: charge every day 10:00-14:00 local time
NZ_BOX = (-47.6, -34.0, 165.8, 178.9)        # lat min, lat max, lon min, lon max (main islands and Stewart Island)


def nearest_region(lat: float, lon: float) -> tuple[str, float]:
    """(name, km) of the closest of the 14 regions (great-circle distance)."""
    best = None
    for name, (la, lo) in config.REGIONS.items():
        p1, p2, dl = np.radians(lat), np.radians(la), np.radians(lo - lon)
        km = 6371 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))
        if best is None or km < best[1]:
            best = (name, float(km))
    return best


# ---------- EV charging rule (same rule as the older charging module, NZ constants) ----------
def plan_charging(forecast_kw, actual_kw) -> tuple[list, float]:
    """Charge each plugged-in hour with the forecast surplus (or as much as needed to finish by the end).

    Returns (kW charged per hour, grid energy caused by the EV in kWh judged with the ACTUAL solar power).
    """
    need, grid, charges, n = config.EV_NEED_KWH, 0.0, [], len(forecast_kw)
    for i in range(n):
        must_now = max(0.0, need - config.CHARGER_KW * (n - i - 1))
        surplus = forecast_kw[i] - config.WORKSHOP_LOAD_KW
        charge = float(np.clip(max(must_now, surplus), 0.0, min(config.CHARGER_KW, need)))
        need -= charge
        charges.append(charge)
        grid += max(0.0, config.WORKSHOP_LOAD_KW + charge - actual_kw[i]) - max(0.0, config.WORKSHOP_LOAD_KW - actual_kw[i])
    return charges, grid


def _num(x, nd=2):
    return None if x is None or pd.isna(x) else round(float(x), nd)


class Service:
    def __init__(self, model, history_long_df: pd.DataFrame, meta: dict | None = None,
                 ahead_models: dict | None = None, ahead_info: dict | None = None, recent_replay: bool = False):
        self.model = model
        self.ahead_models = {1: model, **(ahead_models or {})}             # hours ahead -> model
        self._ahead_info = ahead_info or {}
        self.recent_replay = recent_replay
        self.meta = nzmodel.load_meta() if meta is None else meta
        self.regions = [r for r in config.REGIONS if r in set(history_long_df["region"])]
        self._hist = self._prepare(history_long_df)
        self._live, self._live_at = None, 0.0
        self._live_lock = threading.Lock()
        self._report = None
        self._points: dict = {}                                    # clicked map points, cached per (lat, lon, day)

    # ----- data preparation -----
    def _prepare(self, long_df: pd.DataFrame) -> dict:
        """Features + predictions for every issue hour, plus the raw hourly weather, all indexed by time."""
        feat = build_features(long_df, require_target=False)
        feat["pred_ghi"] = nzmodel.predict(self.model, feat)
        raw = long_df.set_index(["region", "time_utc"]).sort_index()
        return {"feat": feat, "raw": raw}

    def _live_state(self) -> dict:
        new_hour = self._live is not None and pd.Timestamp.now(tz="UTC").floor("h") > self._live["now"] \
            and time.time() - self._live_at > 120
        if self._live is None or time.time() - self._live_at > LIVE_TTL_S or new_hour:
            # One download at a time. While it runs (or if it fails), other requests keep the last good live data.
            if not self._live_lock.acquire(blocking=self._live is None):
                return self._live
            try:
                live = data.fetch_live(self.regions)
                state = self._prepare(live)
                state["long"] = live
                fc = live[live["is_forecast"]]["time_utc"]
                state["now"] = fc.min() - HOUR if len(fc) else live["time_utc"].max()   # latest real hour
                self._live, self._live_at = state, time.time()
            except Exception:
                if self._live is None:
                    raise
                self._live_at = time.time() - LIVE_TTL_S + 120          # keep the old hour, try again in 2 minutes
            finally:
                self._live_lock.release()
        return self._live

    def _state(self, time_utc):
        """(state, issue time t, live?)"""
        if str(time_utc).lower() == "live" or str(time_utc).startswith("ahead:"):
            st = self._live_state()
            return st, st["now"], True
        t = pd.Timestamp(time_utc)
        t = (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).floor("h")
        if self.recent_replay and self._live is not None and self._live["now"] - pd.Timedelta("47h") <= t < self._live["now"]:
            return self._live, t, False                                  # one of the past 24 hours (all real data)
        return self._hist, t, False

    @staticmethod
    def hours_ahead(time_utc) -> int:
        """1 for "live" and replay hours; h for "ahead:h"."""
        s = str(time_utc)
        return int(s.split(":", 1)[1]) if s.startswith("ahead:") else 1

    def ahead_options(self) -> list[dict]:
        """The "Predict next" choices: hours ahead with a model, and how accurate each was on the unseen test period."""
        test = self._ahead_info.get("test", {})
        return [{"hours": h, **test.get(str(h), {})} for h in sorted(self.ahead_models)]

    def _ahead_feat(self, h: int) -> pd.DataFrame:
        """Live features and predictions for the hour ending h hours after each issue hour (cached per live download)."""
        st = self._live_state()
        cache = st.setdefault("ahead", {1: st["feat"]})
        if h not in cache:
            feat = build_features(st["long"], require_target=False, horizon=h)
            feat["pred_ghi"] = nzmodel.predict(self.ahead_models[h], feat)
            cache[h] = feat
        return cache[h]

    def ahead(self, region: str) -> list[dict]:
        """From the latest real hour: the forecast for each of the next hours that has a model (Predict next)."""
        st = self._live_state()
        out = []
        for h in sorted(self.ahead_models):
            key = (region, st["now"] + h * HOUR)
            f = self._ahead_feat(h)
            if key not in f.index:
                continue
            row = f.loc[key]
            start = (st["now"] + (h - 1) * HOUR).tz_convert(config.LOCAL_TZ)
            out.append({"hours": h, "from_local": start.strftime("%H:%M"),
                        "to_local": (start + HOUR).strftime("%H:%M"),
                        "kw": _num(nzmodel.to_kw(row["pred_ghi"], row["temp_t"])),
                        "cloud_cover_pct": _num(row["fc_cloud_target"] if "fc_cloud_target" in row else None, 0)})
        return out

    # ----- times -----
    def replay_window(self) -> str:
        """"past_24h" when Replay shows the last 24 hours of live data, "history" for the saved history file."""
        if self.recent_replay:
            try:
                self._live_state()
                return "past_24h"
            except Exception:
                pass
        return "history"

    def available_times(self) -> list[str]:
        """Replay hours: the past 24 hours (recent_replay), else test-period hours where most regions have daylight."""
        if self.replay_window() == "past_24h":
            now = self._live["now"]
            return [(now - k * HOUR).strftime(ISO) for k in range(24, 0, -1)]
        feat = self._hist["feat"]
        t0, t1 = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS["test"])
        day = daytime_mask(feat).groupby(level="target_time").mean()
        issue = day.index - HOUR
        ok = (day >= 0.5).to_numpy() & (issue >= t0) & (issue < t1)
        if not ok.any():                                           # history outside the test period (e.g. tests)
            ok = (day >= 0.5).to_numpy()
        return [t.strftime(ISO) for t in issue[ok]]

    def default_time(self) -> str:
        """Past 24 hours: the sunniest hour. History: a local midday in the test summer with mixed cloud."""
        if self.replay_window() == "past_24h":
            f, now = self._live["feat"], self._live["now"]
            tm = f.index.get_level_values("target_time")
            sun = f[(tm > now - 24 * HOUR) & (tm <= now)]["pred_ghi"].groupby(level="target_time").mean()
            return (sun.idxmax() - HOUR).strftime(ISO) if len(sun) else (now - 3 * HOUR).strftime(ISO)
        times = pd.DatetimeIndex([pd.Timestamp(t) for t in self.available_times()])
        local = times.tz_convert(config.LOCAL_TZ)
        mid = times[(local.hour == 12) & np.isin(local.month, [12, 1, 2])]
        if len(mid) == 0:
            return self.available_times()[len(times) // 2]
        cloud = self._hist["feat"]["meta_cloud_now"].unstack(level=0)      # target_time x region
        c = cloud.reindex(mid + HOUR)
        score = (c.mean(axis=1) - 50).abs() - 0.5 * c.std(axis=1)
        return mid[int(np.argmin(score.fillna(1e9).to_numpy()))].strftime(ISO)

    # ----- snapshot -----
    def snapshot(self, time_utc) -> list[dict]:
        st, t, live = self._state(time_utc)
        h = self.hours_ahead(time_utc)
        feat = self._ahead_feat(h) if h > 1 else st["feat"]
        out = []
        for r in self.regions:
            lat, lon = config.REGIONS[r]
            row = feat.loc[(r, t + h * HOUR)] if (r, t + h * HOUR) in feat.index else None
            if row is None:
                continue
            raw = st["raw"].loc[(r, t)] if (r, t) in st["raw"].index else None
            temp = _num(row["temp_t"], 1)
            pred = float(nzmodel.to_kw(row["pred_ghi"], row["temp_t"]))
            pers = float(nzmodel.to_kw(max(row["ghi_t"], 0), row["temp_t"])) if pd.notna(row["ghi_t"]) else 0.0
            actual = None if live or pd.isna(row["target_kw"]) else float(row["target_kw"])
            out.append({"name": r, "lat": lat, "lon": lon, "hours_ahead": h,
                        "local_time": t.tz_convert(config.LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                        "cloud_cover_pct": _num(row["cloud_t"], 0), "temp_c": temp,
                        "wind_kmh": _num(row["wind_t"], 1), "ghi_now_wm2": _num(row["ghi_t"], 0),
                        "is_day": bool(row["elev_target"] > 0) if h > 1 else bool(raw["is_day"]) if raw is not None and pd.notna(raw["is_day"]) else bool(row["elev_target"] > 0),
                        "predicted_kw_next": _num(pred), "persistence_kw_next": _num(pers),
                        "actual_kw_next": _num(actual), "generating": bool(pred > 1)})
        return out

    # ----- detail -----
    def detail(self, region: str, time_utc) -> dict:
        st, t, live = self._state(time_utc)
        metrics = self.meta.get("metrics", {}).get("test", {}).get("per_region", {}).get(region, {})
        known = st["now"] if (not live and "now" in st) else None        # past 24 h: what happened up to now
        res = self._day_detail(st["feat"].loc[region].copy(), region, t, live, metrics, known)
        if live:
            res["ahead"] = self.ahead(region)
        return res

    # ----- any clicked point in New Zealand -----
    def point(self, lat: float, lon: float, time_utc) -> dict:
        """Forecast for ANY map point: download that point's weather from Open-Meteo and run the same model.

        The point borrows the nearest region's id (its climate) and uses its own latitude and sun position.
        Its accuracy was not measured separately: the panel shows the nearest region's test error as a guide.
        """
        lat, lon = float(lat), float(lon)
        if not (NZ_BOX[0] <= lat <= NZ_BOX[1] and NZ_BOX[2] <= lon <= NZ_BOX[3]):
            raise ValueError("Please click inside New Zealand.")
        live = str(time_utc).lower() == "live" or str(time_utc).startswith("ahead:")
        st0, t, _ = (None, None, None) if live else self._state(time_utc)
        recent = not live and st0 is self._live                          # one of the past 24 hours: live download
        key = (round(lat, 2), round(lon, 2), "live" if live or recent else t.tz_convert(config.LOCAL_TZ).date())
        st = self._points.get(key)
        if st is None or ((live or recent) and time.time() - st["at"] > LIVE_TTL_S):
            nearest, dist = nearest_region(lat, lon)
            name = f"Point {lat:.2f}, {lon:.2f}"
            if live or recent:
                raw = data.fetch_point(lat, lon, name)
                fc = raw[raw["is_forecast"]]["time_utc"]
                now = fc.min() - HOUR if len(fc) else raw["time_utc"].max()
            else:
                day = t.tz_convert(config.LOCAL_TZ).normalize()
                start = (day - pd.Timedelta("2D")).tz_convert("UTC").strftime("%Y-%m-%d")
                end = (day + pd.Timedelta("2D")).tz_convert("UTC").strftime("%Y-%m-%d")
                raw, now = data.fetch_point(lat, lon, name, start, end), None
            raw = raw.assign(loc_lat=lat, loc_lon=lon, loc_region_id=list(config.REGIONS).index(nearest))
            feat = build_features(raw, require_target=False)
            feat["pred_ghi"] = nzmodel.predict(self.model, feat)
            st = {"feat": feat, "raw": raw.set_index("time_utc"), "name": name, "nearest": nearest, "dist": dist,
                  "now": now, "at": time.time()}
            if len(self._points) > 200:
                self._points.clear()
            self._points[key] = st
        t = st["now"] if live else t
        f = st["feat"].loc[st["name"]].copy()
        if t + HOUR not in f.index:
            raise ValueError("No weather data for this point at this time.")
        row = f.loc[t + HOUR]
        pred = float(nzmodel.to_kw(row["pred_ghi"], row["temp_t"]))
        actual = None if live or pd.isna(row["target_kw"]) or (recent and t + HOUR > st["now"]) else float(row["target_kw"])
        metrics = self.meta.get("metrics", {}).get("test", {}).get("per_region", {}).get(st["nearest"], {})
        res = self._day_detail(f, st["name"], t, live, metrics, st["now"] if recent else None)
        res.update({"name": st["name"], "lat": lat, "lon": lon, "nearest_region": st["nearest"],
                    "distance_km": round(st["dist"], 1), "is_point": True,
                    "local_time": t.tz_convert(config.LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                    "cloud_now_pct": _num(row["cloud_t"], 0), "temp_c": _num(row["temp_t"], 1),
                    "wind_kmh": _num(row["wind_t"], 1), "ghi_now_wm2": _num(row["ghi_t"], 0),
                    "predicted_kw_next": _num(pred), "actual_kw_next": _num(actual), "generating": bool(pred > 1)})
        return res

    def _day_detail(self, f: pd.DataFrame, region: str, t, live: bool, metrics: dict, known_until=None) -> dict:
        """The local day of one location (feature rows indexed by target_time) -> chart lists, EV advice, savings."""
        nxt = f.loc[t + HOUR] if t + HOUR in f.index else None     # the next hour, shown first in the panel
        next_kw = None if nxt is None else float(nzmodel.to_kw(nxt["pred_ghi"], nxt["temp_t"]))
        start = f.index - HOUR                                     # the hour each row forecasts: [start, start+1h)
        local = start.tz_convert(config.LOCAL_TZ)
        day = t.tz_convert(config.LOCAL_TZ).date()                 # local day of the NEXT hour is t's local day
        f = f[local.date == day]
        local = local[local.date == day]
        known = (f.index <= t) if live else (f.index <= known_until) if known_until is not None else np.ones(len(f), bool)
        pred_kw = nzmodel.to_kw(f["pred_ghi"], f["temp_t"])
        pers_kw = nzmodel.to_kw(f["ghi_t"].clip(lower=0), f["temp_t"])
        actual_kw = np.where(known, f["target_kw"], np.nan)

        # EV charging for the plug-in hours of this local day: forecast-driven vs. charge on arrival
        slot = (local.hour >= config.WORK_START) & (local.hour < config.WORK_END)
        n_slots = config.WORK_END - config.WORK_START
        advice = {"charge_now": False, "reason": "No data for this local day."}
        saved = {"grid_kwh_saved": 0.0, "money_saved_nzd": 0.0, "co2_saved_kg": 0.0, "grid_kwh_saved_vs_fixed": 0.0,
                 "money_saved_vs_fixed_nzd": 0.0}
        if slot.sum() == n_slots:
            fc = np.nan_to_num(np.asarray(pred_kw)[slot])
            act = np.nan_to_num(np.where(np.isnan(actual_kw), pred_kw, actual_kw)[slot])   # live: estimate the rest
            charges, grid = plan_charging(fc, act)
            big = config.WORKSHOP_LOAD_KW + config.CHARGER_KW
            _, grid_arrival = plan_charging(np.full(n_slots, big), act)
            noon = np.where((local[slot].hour >= FIXED_WINDOW[0]) & (local[slot].hour < FIXED_WINDOW[1]), big, 0.0)
            _, grid_fixed = plan_charging(noon, act)               # a stronger habit: always charge 10:00-14:00
            kwh = grid_arrival - grid
            saved = {"grid_kwh_saved": kwh, "money_saved_nzd": kwh * config.GRID_PRICE_NZD,
                     "co2_saved_kg": kwh * config.GRID_CO2_KG_PER_KWH, "grid_kwh_saved_vs_fixed": grid_fixed - grid,
                     "money_saved_vs_fixed_nzd": (grid_fixed - grid) * config.GRID_PRICE_NZD}
            advice = self._advice(t, local, slot, np.asarray(pred_kw), charges)
        res = {"region": region, "local_date": str(day), "hours": [int(h) for h in local.hour],
               "actual_kw": [_num(v) for v in actual_kw], "predicted_kw": [_num(v) for v in pred_kw],
               "persistence_kw": [_num(v) for v in pers_kw], "cloud_cover_pct": [_num(v, 0) for v in f["cloud_t"]],
               "predicted_kw_next": _num(next_kw), "next_local_hour": int(t.tz_convert(config.LOCAL_TZ).hour),
               "totals_estimated": bool(live), **advice,
               **{k: round(float(v), 2) for k, v in saved.items()},
               "metrics": metrics}
        return res

    @staticmethod
    def _advice(t, local, slot, pred_kw, charges) -> dict:
        """Advice for the next hour [t, t+1h): does the rule charge the EV in that slot?"""
        h = t.tz_convert(config.LOCAL_TZ)
        label = f"{h.hour:02d}:00"
        if not (config.WORK_START <= h.hour < config.WORK_END):
            return {"charge_now": False, "reason": f"{label} is outside the {config.WORK_START:02d}:00-{config.WORK_END:02d}:00 plug-in window."}
        i = int(np.where(local[slot].hour == h.hour)[0][0])
        kw = float(pred_kw[slot][i])
        c = charges[i]
        if c > 0.05:
            why = "enough solar surplus" if kw - config.WORKSHOP_LOAD_KW > 0.05 else "the EV must finish charging by 18:00"
            return {"charge_now": True, "reason": f"Charge at {c:.1f} kW: forecast solar {kw:.1f} kW, {why}."}
        if sum(charges[:i]) >= config.EV_NEED_KWH - 0.05:
            return {"charge_now": False, "reason": "The EV is already full for today."}
        return {"charge_now": False, "reason": f"Wait: forecast solar {kw:.1f} kW does not cover the {config.WORKSHOP_LOAD_KW:.0f} kW workshop load."}

    # ----- model report (dashboard "Model" view) -----
    def model_report(self) -> dict:
        """Saved at training time in models/nz_model.json; computed from the history if the JSON has no report."""
        if self._report is None:
            rep = self.meta.get("report") or self._compute_report()
            extra = {k: self.meta[k] for k in ("version", "changes", "experiments") if self.meta.get(k)}
            prev = self.meta.get("previous")
            if prev:                                                   # the model this one replaced, numbers only
                extra["previous"] = {"version": prev.get("version"), "note": prev.get("note"),
                                     "val": prev["metrics"]["val"]["overall"]["model"],
                                     "test": prev["metrics"]["test"]["overall"]["model"]}
            self._report = {**rep, **extra} if extra else rep
        return self._report

    def _compute_report(self) -> dict:
        feat = self._hist["feat"]
        feat = feat[feat["target_ghi"].notna()]
        tm = feat.index.get_level_values("target_time")
        span = lambda k: (tm >= pd.Timestamp(config.SPLITS[k][0], tz="UTC")) & (tm < pd.Timestamp(config.SPLITS[k][1], tz="UTC"))
        return nzmodel.build_report(self.model, feat[span("val")], feat[span("test")] if span("test").any() else feat)
