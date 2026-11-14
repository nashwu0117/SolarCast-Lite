"""Time splits, metrics, error analysis and plots. Plots return a Figure (never plt.show())."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from solarcast import config

POWER_BINS = [0, 10, 20, 30, 40, 50]


def split_by_time(feat: pd.DataFrame, splits: dict = config.SPLITS) -> dict:
    """Cut by target time (the index) into half-open [start, end) pieces. No shuffling."""
    out = {}
    for name, (start, end) in splits.items():
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        out[name] = feat[(feat.index >= lo) & (feat.index < hi)]
    return out


def daytime_mask(feat: pd.DataFrame) -> pd.Series:
    """True where the sun is above the daytime threshold at the target hour."""
    return feat["meta_target_elevation"] > config.DAYTIME_ELEVATION_MIN


def regression_metrics(y_true, y_pred, capacity_kw: float = config.CAPACITY_KW) -> dict:
    """MAE and RMSE in kW, nMAE as % of plant capacity, and the number of rows."""
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(err)))
    return {"mae_kw": mae, "rmse_kw": float(np.sqrt(np.mean(err ** 2))),
            "nmae_pct": mae / capacity_kw * 100, "n": int(len(err))}


def _select(feat, predictions, daytime_only):
    """Restrict feat and every prediction to daytime rows (if asked)."""
    if daytime_only:
        feat = feat[daytime_mask(feat)]
    return feat, {name: p.loc[feat.index] for name, p in predictions.items()}


def compare(feat, predictions: dict, daytime_only: bool = True) -> pd.DataFrame:
    """One row per model: MAE, RMSE, nMAE, n. Sorted best (lowest MAE) first."""
    feat, preds = _select(feat, predictions, daytime_only)
    rows = {name: regression_metrics(feat["target"], p) for name, p in preds.items()}
    return pd.DataFrame(rows).T.astype({"n": int}).sort_values("mae_kw")


def grouped_metrics(feat, predictions: dict, by: str, daytime_only: bool = True) -> pd.DataFrame:
    """Metrics per group: by="sky" (cloud cover bins) or by="power_bin" (actual kW bins)."""
    feat, preds = _select(feat, predictions, daytime_only)
    if by == "sky":
        cloud = feat["meta_target_cloud_cover"]
        groups = {g: (cloud >= lo) & (cloud < hi) for g, (lo, hi) in config.SKY_BINS.items()}
    elif by == "power_bin":
        edges = POWER_BINS[:-1] + [np.inf]          # last bin also holds 50 kW itself
        groups = {f"{lo}-{hi}": (feat["target"] >= lo) & (feat["target"] < e)
                  for lo, hi, e in zip(POWER_BINS[:-1], POWER_BINS[1:], edges[1:])}
    else:
        raise ValueError("by must be 'sky' or 'power_bin'")
    rows = []
    for group, mask in groups.items():
        for model, p in preds.items():
            m = regression_metrics(feat.loc[mask, "target"], p[mask]) if mask.any() else \
                {"mae_kw": np.nan, "rmse_kw": np.nan, "nmae_pct": np.nan, "n": 0}
            rows.append({"group": group, "model": model, **m})
    return pd.DataFrame(rows)


def worst_errors(feat, predictions: dict, model: str, k: int = 10) -> pd.DataFrame:
    """The k daytime hours where `model` was furthest off."""
    feat, preds = _select(feat, predictions, True)
    out = pd.DataFrame({"actual_kw": feat["target"], "predicted_kw": preds[model],
                        "cloud_cover_pct": feat["meta_target_cloud_cover"]})
    out["abs_error_kw"] = (out["actual_kw"] - out["predicted_kw"]).abs()
    return out.sort_values("abs_error_kw", ascending=False).head(k)


def plot_actual_vs_predicted(feat, predictions: dict, start=None, end=None) -> Figure:
    """Line chart of actual power and each model's prediction between start and end."""
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    sl = slice(start, end)
    ax.plot(feat["target"].loc[sl], color="black", lw=2, label="Actual")
    for name, p in predictions.items():
        ax.plot(p.loc[sl], lw=1.2, label=name)
    ax.set_ylabel("Power (kW)")
    ax.set_title("Actual vs predicted power")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_error_by_group(grouped_df: pd.DataFrame) -> Figure:
    """Grouped bar chart of MAE (kW) per group, one bar per model."""
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    grouped_df.pivot(index="group", columns="model", values="mae_kw").loc[grouped_df["group"].unique()] \
        .plot.bar(ax=ax, rot=0)
    ax.set_ylabel("MAE (kW)")
    ax.set_title("Error by group")
    return fig
