"""Evaluation tests on small hand-built frames (do not need features.py)."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from solarcast import config, evaluate


def make_feat(idx, target, cloud=50.0, elev=30.0):
    return pd.DataFrame({"target": target, "meta_target_cloud_cover": cloud,
                         "meta_target_elevation": elev}, index=idx)


def test_regression_metrics_hand_computed():
    m = evaluate.regression_metrics([10, 20, 30], [12, 18, 36])   # errors 2, 2, 6
    assert m["mae_kw"] == pytest.approx(10 / 3)
    assert m["rmse_kw"] == pytest.approx(np.sqrt(44 / 3))
    assert m["nmae_pct"] == pytest.approx(m["mae_kw"] / 50 * 100)
    assert m["n"] == 3
    assert "mape" not in m


def test_daytime_mask():
    f = make_feat(pd.date_range("2026-03-01", periods=3, freq="h", tz="UTC"), 1.0)
    f["meta_target_elevation"] = [-3.0, 5.0, 5.1]
    assert evaluate.daytime_mask(f).tolist() == [False, False, True]


def test_split_by_time_half_open_and_disjoint():
    idx = pd.date_range("2024-12-31 22:00", "2026-07-01 02:00", freq="h", tz="UTC", name="target_time")
    f = make_feat(idx, 1.0)
    s = evaluate.split_by_time(f)
    assert set(s) == {"train", "val", "test"}
    assert s["train"].index.min() == pd.Timestamp("2025-01-01", tz="UTC")
    assert s["train"].index.max() == pd.Timestamp("2025-12-31 23:00", tz="UTC")
    assert s["val"].index.min() == pd.Timestamp("2026-01-01", tz="UTC")      # boundary goes to later split
    assert s["test"].index.min() == pd.Timestamp("2026-03-01", tz="UTC")
    assert s["test"].index.max() == pd.Timestamp("2026-06-30 23:00", tz="UTC")
    all_idx = s["train"].index.append(s["val"].index).append(s["test"].index)
    assert all_idx.is_unique


def _frame_and_preds():
    idx = pd.date_range("2026-03-01", periods=6, freq="h", tz="UTC")
    f = make_feat(idx, [5.0, 15.0, 25.0, 35.0, 45.0, 50.0])
    f["meta_target_cloud_cover"] = [10, 40, 80, 20, 60, 100]
    f["meta_target_elevation"] = [30, 30, 30, 30, 30, 1]        # last row is night
    good = f["target"] + 1
    bad = f["target"] + 5
    return f, {"good": good, "bad": bad}


def test_compare_sorted_and_daytime_only():
    f, p = _frame_and_preds()
    t = evaluate.compare(f, p)
    assert list(t.index) == ["good", "bad"]
    assert t.loc["good", "n"] == 5 and t.loc["good", "mae_kw"] == pytest.approx(1)
    assert evaluate.compare(f, p, daytime_only=False).loc["good", "n"] == 6


def test_grouped_metrics_sky():
    f, p = _frame_and_preds()
    g = evaluate.grouped_metrics(f, p, by="sky")
    assert list(g.columns) == ["group", "model", "mae_kw", "rmse_kw", "nmae_pct", "n"]
    assert set(g["group"]) == set(config.SKY_BINS)
    row = g[(g.group == "clear") & (g.model == "good")].iloc[0]
    assert row["n"] == 2                       # cloud 10 and 20
    assert g[(g.group == "overcast") & (g.model == "bad")]["n"].iloc[0] == 1   # night row excluded


def test_grouped_metrics_power_bin():
    f, p = _frame_and_preds()
    g = evaluate.grouped_metrics(f, p, by="power_bin", daytime_only=False)
    assert list(g["group"].unique()) == ["0-10", "10-20", "20-30", "30-40", "40-50"]
    n = g[g.model == "good"].set_index("group")["n"]
    assert n.tolist() == [1, 1, 1, 1, 2]       # 50 kW lands in the top bin
    with pytest.raises(ValueError):
        evaluate.grouped_metrics(f, p, by="nope")


def test_worst_errors_and_plots():
    f, p = _frame_and_preds()
    w = evaluate.worst_errors(f, p, "bad", k=2)
    assert len(w) == 2 and (w["abs_error_kw"] == 5).all()
    fig = evaluate.plot_actual_vs_predicted(f, p)
    fig2 = evaluate.plot_error_by_group(evaluate.grouped_metrics(f, p, by="sky"))
    assert isinstance(fig, Figure) and isinstance(fig2, Figure)
