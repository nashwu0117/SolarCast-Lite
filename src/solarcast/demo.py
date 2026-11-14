"""Gradio demo: replay one-hour-ahead forecasts on held-out test hours.

`predict_at` is plain Python (easy to test); `build_demo` imports gradio lazily so the
rest of the package works without it.
"""
from __future__ import annotations

import pandas as pd

from . import config

NOTE = "Historical replay on held-out test data. Prototype."
SITE_CHOICES = ["A", "B"]


def _prediction_at(result, model_name: str, target_time: pd.Timestamp) -> float:
    """Look up a model's prediction for the target hour in the test (else val) split."""
    for split in ("test", "val"):
        preds = result.predictions.get(split, {}).get(model_name)
        if preds is not None and target_time in preds.index:
            return float(preds.loc[target_time])
    raise KeyError(f"No '{model_name}' prediction for {target_time}")


def predict_at(result, target_time) -> dict:
    """Everything the demo shows for one target hour (UTC). Issue time is target_time - 1 h."""
    target_time = pd.Timestamp(target_time)
    if target_time.tzinfo is None:
        target_time = target_time.tz_localize("UTC")
    else:
        target_time = target_time.tz_convert("UTC")
    issue_time = target_time - pd.Timedelta(hours=1)

    row = result.feat.loc[target_time]
    actual = float(row["target"])
    model_b = _prediction_at(result, "Model B", target_time)
    model_a = _prediction_at(result, "Model A", target_time)
    persistence = _prediction_at(result, "Persistence", target_time)
    previous_day = _prediction_at(result, "Previous-day same hour", target_time)

    recent = result.df[config.TARGET_COL].loc[issue_time - pd.Timedelta(hours=5): issue_time]
    weather_now = {c: float(result.df.loc[issue_time, c]) for c in config.WEATHER_COLS}

    return {
        "target_time": target_time,
        "issue_time": issue_time,
        "model_b_kw": model_b,
        "model_a_kw": model_a,
        "persistence_kw": persistence,
        "previous_day_kw": previous_day,
        "actual_kw": actual,
        "abs_error_kw": abs(model_b - actual),
        "capacity_factor": actual / config.CAPACITY_KW,
        "recent_power": recent,
        "weather_now": weather_now,
    }


# ---------------------------------------------------------------------------- helpers

def _daytime_test_times(result) -> pd.DatetimeIndex:
    test = result.splits["test"]
    mask = test["meta_target_elevation"] > config.DAYTIME_ELEVATION_MIN
    return test.index[mask.to_numpy()]


def _label(t: pd.Timestamp) -> str:
    local = t.tz_convert(config.LOCAL_TZ)
    return f"{local:%Y-%m-%d %H:%M} local ({t:%H:%M} UTC)"


def _choices(result):
    times = _daytime_test_times(result)
    return [_label(t) for t in times], {_label(t): t for t in times}


def _default_label(result, labels, mapping) -> str:
    """A mid-day hour on a partly cloudy day (cloud cover 30-70 %) near the middle of the test split."""
    test = result.splits["test"]
    times = _daytime_test_times(result)
    local_hour = times.tz_convert(config.LOCAL_TZ).hour
    cloud = test.loc[times, "meta_target_cloud_cover"].to_numpy()
    lo, hi = config.SKY_BINS["partly_cloudy"]
    ok = [t for t, h, c in zip(times, local_hour, cloud) if h == 13 and lo <= c < hi]
    if ok:
        return _label(ok[len(ok) // 2])
    return labels[len(labels) // 2]


def _fig_recent(info):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    rec = info["recent_power"]
    ax.plot(rec.index.tz_convert(config.LOCAL_TZ), rec.values, marker="o", color="tab:blue",
            label="Measured power")
    ax.scatter([info["target_time"].tz_convert(config.LOCAL_TZ)], [info["actual_kw"]], marker="*",
               s=120, color="tab:green", label="Next hour (actual)", zorder=3)
    ax.scatter([info["target_time"].tz_convert(config.LOCAL_TZ)], [info["model_b_kw"]], marker="x",
               s=70, color="tab:red", label="Next hour (Model B)", zorder=3)
    ax.set_ylabel("Power (kW)")
    ax.set_title("Last 6 hours before issue time, and the next hour")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _fig_day(result, info):
    import matplotlib.pyplot as plt

    tz = config.LOCAL_TZ
    day = info["target_time"].tz_convert(tz).date()
    test = result.splits["test"]
    local_dates = test.index.tz_convert(tz).date
    day_idx = test.index[local_dates == day]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(day_idx.tz_convert(tz), test.loc[day_idx, "target"], color="black", marker="o",
            ms=3, label="Actual")
    for name, color in (("Model B", "tab:red"), ("Model A", "tab:orange")):
        preds = result.predictions["test"][name].reindex(day_idx)
        ax.plot(day_idx.tz_convert(tz), preds.values, color=color, marker=".", label=name)
    ax.axvline(info["target_time"].tz_convert(tz), color="grey", ls="--", label="Selected hour")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Actual vs predicted on {day} (local time)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _weather_md(info) -> str:
    w = info["weather_now"]
    return (
        f"**Weather at issue time** ({info['issue_time']:%Y-%m-%d %H:%M} UTC, Open-Meteo reanalysis)\n\n"
        f"- Cloud cover: {w['cloud_cover_pct']:.0f} %\n"
        f"- Temperature: {w['temp_c']:.1f} C\n"
        f"- Humidity: {w['humidity_pct']:.0f} %\n"
        f"- Wind: {w['wind_speed_kmh']:.1f} km/h\n"
        f"- Precipitation: {w['precip_mm']:.1f} mm\n"
        f"- GHI / DNI / DHI: {w['ghi_wm2']:.0f} / {w['dni_wm2']:.0f} / {w['dhi_wm2']:.0f} W/m2"
    )


def _result_table(info) -> pd.DataFrame:
    rows = [
        ("Model B prediction (power + weather)", f"{info['model_b_kw']:.2f} kW"),
        ("Model A prediction (power only)", f"{info['model_a_kw']:.2f} kW"),
        ("Persistence", f"{info['persistence_kw']:.2f} kW"),
        ("Previous-day same hour", f"{info['previous_day_kw']:.2f} kW"),
        ("Actual", f"{info['actual_kw']:.2f} kW"),
        ("Absolute error of Model B", f"{info['abs_error_kw']:.2f} kW"),
        ("Capacity factor (actual / capacity)", f"{info['capacity_factor'] * 100:.1f} %"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def _metrics_table(result) -> pd.DataFrame:
    m = result.metrics["test"].copy()
    if m.index.name or not isinstance(m.index, pd.RangeIndex):
        m = m.reset_index().rename(columns={"index": "model"})
    return m.round(3)


# ------------------------------------------------------------------------------- UI

def build_demo(results: dict):
    """Build (but do not launch) the Gradio app. `results` maps site letter -> ExperimentResult."""
    import gradio as gr

    sites = [s for s in SITE_CHOICES if s in results] or list(results)
    cache = {s: _choices(results[s]) for s in sites}

    def render(site, label):
        result = results[site]
        labels, mapping = cache[site]
        if label not in mapping:
            label = _default_label(result, labels, mapping)
        info = predict_at(result, mapping[label])
        return (_fig_recent(info), _weather_md(info), _result_table(info),
                _fig_day(result, info), _metrics_table(result))

    def on_site(site):
        labels, mapping = cache[site]
        value = _default_label(results[site], labels, mapping)
        return gr.Dropdown(choices=labels, value=value)

    with gr.Blocks(title="SolarCast Lite") as demo:
        gr.Markdown("# SolarCast Lite: one-hour-ahead solar power forecast\n"
                    f"_{NOTE}_ Weather inputs are Open-Meteo reanalysis, not a live forecast.")
        with gr.Row():
            site = gr.Radio(sites, value=sites[0], label="Site")
            labels0, map0 = cache[sites[0]]
            when = gr.Dropdown(labels0, value=_default_label(results[sites[0]], labels0, map0),
                               label="Target hour (daytime, test split)", filterable=True)
        with gr.Row():
            recent_plot = gr.Plot(label="Recent power")
            weather = gr.Markdown()
        table = gr.Dataframe(label="Prediction vs baselines vs actual", interactive=False)
        day_plot = gr.Plot(label="Actual vs predicted, whole day")
        metrics = gr.Dataframe(label="Test-set metrics (daytime hours)", interactive=False)
        gr.Markdown(f"*{NOTE}*")

        outputs = [recent_plot, weather, table, day_plot, metrics]
        site.change(on_site, site, when)
        when.change(render, [site, when], outputs)
        demo.load(render, [site, when], outputs)
    return demo
