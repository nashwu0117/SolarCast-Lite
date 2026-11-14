"""What the web app calls: replay any past hour of the history, or go live, for all regions.

`time_utc` is the ISSUE hour t (UTC ISO string, e.g. "2026-01-15T00:00:00Z"): we know everything up to the end of
hour t and forecast the hour [t, t+1h). "live" means the latest real hour from the Open-Meteo forecast API.
"""
import time

import numpy as np
import pandas as pd

from solarcast.nz import config, data, model as nzmodel
from solarcast.nz.features import FEATURES, build_features, daytime_mask

HOUR = pd.Timedelta("1h")
ISO = "%Y-%m-%dT%H:%M:%SZ"
LIVE_TTL_S = 600


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
    def __init__(self, model, history_long_df: pd.DataFrame, meta: dict | None = None):
        self.model = model
        self.meta = nzmodel.load_meta() if meta is None else meta
        self.regions = [r for r in config.REGIONS if r in set(history_long_df["region"])]
        self._hist = self._prepare(history_long_df)
        self._live, self._live_at = None, 0.0
        self._report = None

    # ----- data preparation -----
    def _prepare(self, long_df: pd.DataFrame) -> dict:
        """Features + predictions for every issue hour, plus the raw hourly weather, all indexed by time."""
        feat = build_features(long_df, require_target=False)
        feat["pred_ghi"] = nzmodel.predict(self.model, feat)
        raw = long_df.set_index(["region", "time_utc"]).sort_index()
        return {"feat": feat, "raw": raw}

    def _live_state(self) -> dict:
        if self._live is None or time.time() - self._live_at > LIVE_TTL_S:
            live = data.fetch_live(self.regions)
            state = self._prepare(live)
            fc = live[live["is_forecast"]]["time_utc"]
            state["now"] = fc.min() - HOUR if len(fc) else live["time_utc"].max()   # latest real hour
            self._live, self._live_at = state, time.time()
        return self._live

    def _state(self, time_utc):
        """(state, issue time t, live?)"""
        if str(time_utc).lower() == "live":
            st = self._live_state()
            return st, st["now"], True
        t = pd.Timestamp(time_utc)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return self._hist, t.floor("h"), False

    # ----- times -----
    def available_times(self) -> list[str]:
        """Replay hours in the test period where it is daytime in most regions (so the map is not all dark)."""
        feat = self._hist["feat"]
        t0, t1 = (pd.Timestamp(x, tz="UTC") for x in config.SPLITS["test"])
        day = daytime_mask(feat).groupby(level="target_time").mean()
        issue = day.index - HOUR
        ok = (day >= 0.5).to_numpy() & (issue >= t0) & (issue < t1)
        if not ok.any():                                           # history outside the test period (e.g. tests)
            ok = (day >= 0.5).to_numpy()
        return [t.strftime(ISO) for t in issue[ok]]

    def default_time(self) -> str:
        """A local midday in the test summer with mixed cloud (mean cloud near 50 % and large spread)."""
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
        out = []
        for r in self.regions:
            lat, lon = config.REGIONS[r]
            row = st["feat"].loc[(r, t + HOUR)] if (r, t + HOUR) in st["feat"].index else None
            if row is None:
                continue
            raw = st["raw"].loc[(r, t)] if (r, t) in st["raw"].index else None
            temp = _num(row["temp_t"], 1)
            pred = float(nzmodel.to_kw(row["pred_ghi"], row["temp_t"]))
            pers = float(nzmodel.to_kw(max(row["ghi_t"], 0), row["temp_t"])) if pd.notna(row["ghi_t"]) else 0.0
            actual = None if live or pd.isna(row["target_kw"]) else float(row["target_kw"])
            out.append({"name": r, "lat": lat, "lon": lon,
                        "local_time": t.tz_convert(config.LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                        "cloud_cover_pct": _num(row["cloud_t"], 0), "temp_c": temp,
                        "wind_kmh": _num(row["wind_t"], 1), "ghi_now_wm2": _num(row["ghi_t"], 0),
                        "is_day": bool(raw["is_day"]) if raw is not None and pd.notna(raw["is_day"]) else bool(row["elev_target"] > 0),
                        "predicted_kw_next": _num(pred), "persistence_kw_next": _num(pers),
                        "actual_kw_next": _num(actual), "generating": bool(pred > 1)})
        return out

    # ----- detail -----
    def detail(self, region: str, time_utc) -> dict:
        st, t, live = self._state(time_utc)
        f = st["feat"].loc[region].copy()                          # indexed by target_time
        start = f.index - HOUR                                     # the hour each row forecasts: [start, start+1h)
        local = start.tz_convert(config.LOCAL_TZ)
        day = t.tz_convert(config.LOCAL_TZ).date()                 # local day of the NEXT hour is t's local day
        f = f[local.date == day]
        local = local[local.date == day]
        known = (f.index <= t) if live else np.ones(len(f), bool)   # live: no actuals for hours not yet over
        pred_kw = nzmodel.to_kw(f["pred_ghi"], f["temp_t"])
        pers_kw = nzmodel.to_kw(f["ghi_t"].clip(lower=0), f["temp_t"])
        actual_kw = np.where(known, f["target_kw"], np.nan)

        # EV charging for the plug-in hours of this local day: forecast-driven vs. charge on arrival
        slot = (local.hour >= config.WORK_START) & (local.hour < config.WORK_END)
        n_slots = config.WORK_END - config.WORK_START
        advice = {"charge_now": False, "reason": "No data for this local day."}
        saved = {"grid_kwh_saved": 0.0, "money_saved_nzd": 0.0, "co2_saved_kg": 0.0}
        if slot.sum() == n_slots:
            fc = np.nan_to_num(np.asarray(pred_kw)[slot])
            act = np.nan_to_num(np.where(np.isnan(actual_kw), pred_kw, actual_kw)[slot])   # live: estimate the rest
            charges, grid = plan_charging(fc, act)
            _, grid_arrival = plan_charging(np.full(n_slots, config.WORKSHOP_LOAD_KW + config.CHARGER_KW), act)
            kwh = grid_arrival - grid
            saved = {"grid_kwh_saved": kwh, "money_saved_nzd": kwh * config.GRID_PRICE_NZD,
                     "co2_saved_kg": kwh * config.GRID_CO2_KG_PER_KWH}
            advice = self._advice(t, local, slot, np.asarray(pred_kw), charges)
        res = {"region": region, "local_date": str(day), "hours": [int(h) for h in local.hour],
               "actual_kw": [_num(v) for v in actual_kw], "predicted_kw": [_num(v) for v in pred_kw],
               "persistence_kw": [_num(v) for v in pers_kw], "cloud_cover_pct": [_num(v, 0) for v in f["cloud_t"]],
               "totals_estimated": bool(live), **advice,
               **{k: round(float(v), 2) for k, v in saved.items()},
               "metrics": self.meta.get("metrics", {}).get("test", {}).get("per_region", {}).get(region, {})}
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
