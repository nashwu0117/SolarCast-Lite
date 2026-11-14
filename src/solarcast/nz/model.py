"""Baselines, the SolarCast model (v4: raw-GHI + clear-sky-index gradient boosting, averaged), metrics, saving/loading."""
import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from solarcast.nz import config
from solarcast.nz.data import pv_power_kw
from solarcast.nz.ensemble import DEFAULT_PARAMS, SolarEnsemble      # noqa: F401  (DEFAULT_PARAMS re-exported)
from solarcast.nz.features import FEATURES, daytime_mask

MODEL_NAMES = {"model": "SolarCast model", "persistence": "Persistence", "clear_sky": "Clear-sky persistence"}


# ---------- baselines (predict next-hour GHI in W/m2) ----------
def persistence(feat: pd.DataFrame) -> pd.Series:
    """Next hour looks like the hour that just ended."""
    return feat["ghi_t"].clip(lower=0).rename("persistence")


def clear_sky_persistence(feat: pd.DataFrame) -> pd.Series:
    """Keep the current clear-sky index (GHI / clear-sky GHI) and scale it by the target hour's clear-sky GHI.

    Near dawn/dusk the current index is undefined, so the last 24 h clearness is used instead.
    """
    kt = feat["kt_t"].fillna(feat["kt_roll24"]).fillna(0.5)
    return (kt * feat["cs_target"]).clip(lower=0).rename("clear_sky_persistence")


# ---------- model ----------
V1_PARAMS = {"learning_rate": 0.03, "max_iter": 800, "max_leaf_nodes": 63, "min_samples_leaf": 100,
             "loss": "absolute_error"}          # the tuned v1 settings, reused unchanged by v4 (loss applies to the raw model)


def train(feat_train: pd.DataFrame, **params) -> SolarEnsemble:
    """Fit the v4 model (one pooled ensemble for all regions) on a feature frame. Night rows are skipped: GHI is 0."""
    return SolarEnsemble(FEATURES, {**V1_PARAMS, **params}).fit(feat_train)


def predict(model, feat: pd.DataFrame) -> pd.Series:
    """Next-hour GHI in W/m2, clipped at 0; exactly 0 when the sun is down."""
    ghi = np.where(feat["elev_target"] > 0, np.clip(model.predict(feat[FEATURES]), 0, None), 0.0)
    return pd.Series(ghi, index=feat.index, name="predicted_ghi")


def to_kw(ghi, temp_c) -> np.ndarray:
    """GHI (W/m2) and air temperature -> PV power (kW) of the 50 kW system."""
    return pv_power_kw(ghi, temp_c)


# ---------- metrics ----------
def _err(feat: pd.DataFrame, pred: pd.Series) -> dict:
    e_w = feat["target_ghi"].to_numpy() - pred.to_numpy()
    e_kw = feat["target_kw"].to_numpy() - to_kw(pred.to_numpy(), feat["temp_t"].to_numpy())
    return {"mae_wm2": float(np.mean(np.abs(e_w))), "rmse_wm2": float(np.sqrt(np.mean(e_w ** 2))),
            "mae_kw": float(np.mean(np.abs(e_kw))), "rmse_kw": float(np.sqrt(np.mean(e_kw ** 2))), "n": int(len(e_w))}


def evaluate(feat: pd.DataFrame, preds: dict) -> dict:
    """Daytime MAE/RMSE in W/m2 and kW: {"overall": {name: m}, "per_region": {region: {name: m}}}."""
    keep = daytime_mask(feat) & feat["target_ghi"].notna()
    feat = feat[keep]
    preds = {k: v.loc[feat.index] for k, v in preds.items()}
    out = {"overall": {k: _err(feat, p) for k, p in preds.items()}, "per_region": {}}
    regions = feat.index.get_level_values("region")
    for r in dict.fromkeys(regions):
        m = regions == r
        out["per_region"][r] = {k: _err(feat[m], p[m]) for k, p in preds.items()}
    return out


def all_predictions(model, feat: pd.DataFrame) -> dict:
    return {"model": predict(model, feat), "persistence": persistence(feat), "clear_sky": clear_sky_persistence(feat)}


# ---------- report for the dashboard "Model" view ----------
def build_report(model, feat_val: pd.DataFrame, feat_test: pd.DataFrame, max_rows: int = 20000) -> dict:
    """Numbers for the web app: test metrics, permutation importance on VAL, and one sample week."""
    ev = evaluate(feat_test, all_predictions(model, feat_test))
    overall = [{"model": MODEL_NAMES[k], "mae_kw": m["mae_kw"], "mae_wm2": m["mae_wm2"], "rmse_kw": m["rmse_kw"]}
               for k, m in ev["overall"].items()]
    per_region = [{"region": r, "model": MODEL_NAMES[k], "mae_kw": m["mae_kw"]}
                  for r, d in ev["per_region"].items() for k, m in d.items()]

    val = feat_val[daytime_mask(feat_val) & feat_val["target_ghi"].notna()]
    val = val.sample(min(max_rows, len(val)), random_state=0) if len(val) else val
    importance = []
    if len(val):
        pi = permutation_importance(model, val[FEATURES], val["target_ghi"], scoring="neg_mean_absolute_error",
                                    n_repeats=3, random_state=0)
        top = np.argsort(pi.importances_mean)[::-1][:12]
        importance = [{"feature": FEATURES[i], "importance": float(pi.importances_mean[i])} for i in top]

    return {"periods": {k: [a, b] for k, (a, b) in config.SPLITS.items()}, "overall": overall,
            "per_region": per_region, "importance": importance, "sample": _sample_week(model, feat_test)}


def _sample_week(model, feat_test: pd.DataFrame) -> dict:
    """One week (summer if available) in the region whose test-period cloud cover is closest to 50 %."""
    if feat_test.empty:
        return {}
    cloud = feat_test.groupby(level="region")["meta_cloud_now"].mean()
    region = (cloud - 50).abs().idxmin()
    rf = feat_test.loc[region]
    starts = pd.date_range(rf.index.min().normalize(), rf.index.max(), freq="7D")
    summer = [s for s in starts if s.month in (12, 1, 2)] or list(starts)
    weeks = [(rf[(rf.index >= s) & (rf.index < s + pd.Timedelta("7D"))], s) for s in summer]
    weeks = [w for w, _ in weeks if len(w) >= 160]
    week = max(weeks, key=lambda w: w["meta_cloud_now"].std()) if weeks else rf.iloc[:168]
    kw_pred = to_kw(predict(model, week).to_numpy(), week["temp_t"].to_numpy())
    kw_pers = to_kw(persistence(week).to_numpy(), week["temp_t"].to_numpy())
    return {"region": region, "times": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in week.index],
            "actual_kw": [round(float(v), 2) for v in week["target_kw"]],
            "predicted_kw": [round(float(v), 2) for v in kw_pred],
            "persistence_kw": [round(float(v), 2) for v in kw_pers]}


# ---------- save / load ----------
def save(model, metrics: dict, extra: dict | None = None, model_path=None, meta_path=None) -> None:
    """Write the model (joblib) and a JSON file with versions, features, periods and metrics."""
    model_path, meta_path = model_path or config.MODEL_PATH, meta_path or config.MODEL_META
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    meta = {"sklearn_version": sklearn.__version__, "features": FEATURES, "periods": config.SPLITS,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "metrics": metrics, **(extra or {})}
    meta_path.write_text(json.dumps(meta, indent=2))


def load_meta(meta_path=None) -> dict:
    try:
        return json.loads((meta_path or config.MODEL_META).read_text())
    except (OSError, ValueError):
        return {}


def load(model_path=None, meta_path=None):
    """The saved model, or None if it is missing, unreadable or saved with another scikit-learn version."""
    model_path = model_path or config.MODEL_PATH
    try:
        meta = load_meta(meta_path)
        if meta and meta.get("sklearn_version") != sklearn.__version__:
            warnings.warn("saved model used scikit-learn %s, now %s: retrain" % (meta.get("sklearn_version"), sklearn.__version__))
            return None
        return joblib.load(model_path)
    except Exception as exc:                     # noqa: BLE001  any failure means "retrain"
        warnings.warn(f"could not load model: {exc}")
        return None
