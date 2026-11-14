"""Build notebooks/SolarCast_Lite_Submission.ipynb, the single notebook judges open.

It is self-contained: a hidden first cell writes the `solarcast` package (copied from src/ at build time) into
the Colab VM, so the notebook needs nothing from this repository at run time, only the public dataset.
Rebuild after changing src/:  python scripts/make_submission_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "solarcast"
OUT = ROOT / "notebooks" / "SolarCast_Lite_Submission.ipynb"
MODULES = ["__init__", "config", "data", "features", "baselines", "models", "evaluate", "pipeline", "charging", "demo"]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text, hidden=False):
    cell = nbf.v4.new_code_cell(text.strip())
    if hidden:
        cell.metadata["cellView"] = "form"  # Colab shows only the title, code collapsed
    return cell


def package_cell():
    files = {name: (PKG / f"{name}.py").read_text() for name in MODULES}
    lines = ["# @title Install the SolarCast package (code hidden; full source: github.com/nashwu0117/SolarCast-Lite)",
             "import sys",
             "from pathlib import Path",
             "",
             "PKG_DIR = Path('solarcast_src') / 'src' / 'solarcast'",
             "PKG_DIR.mkdir(parents=True, exist_ok=True)",
             "FILES = {"]
    lines += [f"    {name!r}: {src!r}," for name, src in files.items()]
    lines += ["}",
              "for name, src in FILES.items():",
              "    (PKG_DIR / f'{name}.py').write_text(src)",
              "sys.path.insert(0, str(PKG_DIR.parent))",
              "print('solarcast package ready:', ', '.join(FILES))"]
    return code("\n".join(lines), hidden=True)


DASHBOARD = '''# @title Optional: open the interactive dashboard
RUN_DASHBOARD = False  # @param {type:"boolean"}
# Tick the box (or set True) and run this cell. It starts the Gradio dashboard and publishes it through a free
# Cloudflare quick tunnel (https://....trycloudflare.com). The link works while this Colab runtime is running.
# If the tunnel fails, it falls back to Gradio's own share link.
if RUN_DASHBOARD:
    import importlib.util, os, re, subprocess, sys, time, urllib.request
    if importlib.util.find_spec("gradio") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
    from solarcast.demo import build_demo

    app = build_demo({"A": res_a, "B": res_b})
    app.launch(server_name="127.0.0.1", server_port=7860, prevent_thread_lock=True, share=False)
    url = None
    try:
        if not os.path.exists("cloudflared"):
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                "cloudflared")
            os.chmod("cloudflared", 0o755)
        tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:7860"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + 60
        while url is None and time.time() < deadline:
            line = tunnel.stderr.readline()
            match = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
            url = match.group(0) if match else None
    except Exception as exc:
        print("Cloudflare tunnel failed:", exc)
    if url:
        print("Dashboard (public link, valid while this runtime runs):", url)
    else:
        app.close()
        build_demo({"A": res_a, "B": res_b}).launch(share=True)
else:
    print("Dashboard is off. Tick RUN_DASHBOARD above and run this cell to open it.")'''


def main():
    cells = [
        md("""# SolarCast Lite: when should a small workshop charge its EV?
**Veritas AI International High School Hackathon 2026 · Track: AI + Climate**

**The user.** The owner of a small workshop with a 50 kW rooftop solar system and an electric vehicle that needs
about 40 kWh every working day. Every hour they face one question: *charge now, or wait for more sun?*

**What we built.** (1) **SolarCast**, a machine-learning model that forecasts the next hour's solar power from recent
power, the sun's position and the weather; (2) a charging rule that uses the forecast to charge when solar power
is left over; (3) a day-by-day replay of four months the model never saw, measuring grid electricity, money and
CO2 saved.

**How to run.** Runtime → **Run all**. Everything (data download, training, evaluation) runs here in about a
minute on a free CPU runtime. Results are a historical replay on held-out data.

*This project is a prototype and is not financial or energy advice.*"""),
        package_cell(),
        md("""## 1. The data
**Two-Site Solar PV Inverter & Weather Dataset (Romania)** (CC BY 4.0): real hourly output of two separate 50 kW
solar systems (Site A: Jan 2025-Jun 2026, Site B: May 2025-May 2026), joined with hourly weather (Open-Meteo) and
the sun's position. It downloads from the dataset's public GitHub. Night hours are real zeros; hours the inverter
did not record stay missing (never filled with 0)."""),
        code("""import matplotlib.pyplot as plt
import pandas as pd

from solarcast import config, data

df_a = data.load_site("A")
print(pd.Series(data.data_quality_report(df_a)).to_string())
df_a.loc["2025-06-10":"2025-06-14", config.TARGET_COL].plot(figsize=(10, 3), ylabel="kW",
                                                            title="Site A, five days in June 2025");"""),
        md("""## 2. The task, and the rule that keeps it honest
At the end of hour *t* we predict the power of hour *t+1*. The model may only use what is known at that moment:
power and weather up to hour *t*, plus the sun's position for hour *t+1* (astronomy, known in advance). It never
sees the future. An automated test changes every value from hour *t+1* onward and checks that no input changes.

We split **by time**, never at random: train on 2025, choose settings on Jan-Feb 2026, and report on
**Mar-Jun 2026**, which the model never saw."""),
        code("""from solarcast.features import FEATURE_GROUPS

for group, cols in FEATURE_GROUPS.items():
    print(f"{group:10s}", ", ".join(cols))"""),
        md("""## 3. Baselines and models
- **Persistence:** next hour = this hour. Simple, and hard to beat.
- **Previous-day same hour:** next hour = the same hour yesterday.
- **Model A:** gradient-boosted trees (scikit-learn) on power history and the sun's position.
- **Model B:** Model A plus the weather (cloud cover, sunlight, temperature, wind, rain).

Scores are for daytime hours (sun more than 5° above the horizon). MAE is the average error in kW; nMAE is the
same error as a percentage of the 50 kW capacity."""),
        code("""from solarcast.pipeline import run_experiment

res_a = run_experiment("A")
res_b = run_experiment("B")
for name, res in [("Site A", res_a), ("Site B", res_b)]:
    print(f"{name}, test period Mar-Jun 2026 (never seen in training)")
    display(res.metrics["test"].round(2))"""),
        code("""from solarcast import evaluate

fig = evaluate.plot_actual_vs_predicted(res_a.splits["test"], res_a.predictions["test"],
                                        start="2026-05-18", end="2026-05-25")
fig"""),
        md("""## 4. Where does it fail?
Errors are largest on partly cloudy and overcast hours, when clouds move in or clear within the hour. The worst
single hours are below: a forecast made at the end of one hour cannot know that a cloud is about to arrive."""),
        code("""by_sky = evaluate.grouped_metrics(res_a.splits["test"], res_a.predictions["test"], by="sky")
display(by_sky.pivot(index="group", columns="model", values="mae_kw").round(2))
display(evaluate.worst_errors(res_a.splits["test"], res_a.predictions["test"], "Model B", k=5).round(1))"""),
        md("""## 5. Our addition: from a forecast to a decision
The brief asks what decision a forecast supports. Ours: **when should the workshop charge its EV?**

Assumptions (synthetic, ours): the EV needs 40 kWh a day, an 11 kW charger, plugged in 08:00-18:00, workshop load
6 kW, grid price EUR 0.25/kWh, 0.26 kg CO2/kWh. Each hour one fixed rule decides:
`charge = max(what must be charged now to finish on time, forecast - workshop load)`, up to 11 kW.
The cost is then judged with the **actual** solar power. Only the forecast changes between strategies;
*Perfect forecast* (the real future) is the upper bound. Nothing was tuned on the test period."""),
        code("""from solarcast import charging

COLS = ["days", "ev_grid_kwh", "solar_share_pct", "grid_kwh_saved", "money_saved_eur", "co2_saved_kg", "pct_of_perfect"]
for name, res in [("Site A", res_a), ("Site B", res_b)]:
    preds = {k: v for k, v in res.predictions["test"].items() if k != "Previous-day same hour"}
    _, summary = charging.simulate(res.splits["test"], preds)
    print(f"{name}: EV charging over the test period")
    display(summary[COLS])"""),
        md("""**What we found.**
1. Planning with *any* forecast cuts the grid electricity the EV uses by roughly 70-80% compared with charging on
   arrival.
2. **Surprise:** in spring and summer the simple persistence forecast makes *better* charging decisions than our
   model, although our model's forecasts are about twice as accurate. The table below shows why. Persistence lags
   behind the sunrise, so it under-forecasts mornings and waits, and in summer waiting is free because plenty of
   sun follows. Our model is unbiased, so on borderline morning hours it over-forecasts about half the time and the
   EV draws grid power.
3. We tried a safety margin, chosen on the winter validation months only. There the best margin was zero
   (in winter waiting is costly), so it could not fix summer.

**Lesson:** a more accurate forecast is not automatically a more useful one. The rule also needs to know how much
sun is still coming later today, so our next step is a forecast for the rest of the day."""),
        code("""slots = charging.daily_slots(res_a.splits["test"])
bias = pd.DataFrame({name: res_a.predictions["test"][name].reindex(slots.index) - slots["actual_kw"]
                     for name in ("Persistence", "Model B")})
bias["local hour"] = slots["hour"]
print("Site A test period: average forecast bias (forecast - actual, kW) by hour")
display(bias.groupby("local hour").mean().round(1))"""),
        md("""## 6. Does it work on a different solar system?
Site A's models, trained only on Site A, scored on Site B's test period without retraining. Both sites are in
the same climate region, so this is not evidence for other climates."""),
        code("""from solarcast.pipeline import cross_site

display(cross_site(res_a, res_b).round(2))"""),
        md("""## 7. Interactive dashboard (optional)
Pick a site and an hour from the test period to see the forecast, the baselines and what actually happened."""),
        code(DASHBOARD, hidden=True),
        md("""## Limitations
- **Historical replay.** We replay past hours; this is not a live service.
- **Weather inputs are reanalysis** (what the weather was), not live forecasts. We tested archived weather
  forecasts as inputs on the validation months and they did not help.
- **One climate region** (two plants in Romania, about 18 months of data).
- **The workshop and EV are synthetic assumptions**, and the charging rule is simple.

## Credits
- Dataset: *Two-Site Solar PV Inverter & Weather Dataset (Romania)*,
  https://github.com/upb-nlp/PV-Generation-Dataset, CC BY 4.0 (includes Open-Meteo weather; solar position by pvlib).
- Libraries: scikit-learn, pandas, NumPy, Matplotlib, Gradio. Dashboard tunnel: Cloudflare `cloudflared`.
- AI tools: Claude Code helped write code, run experiments on Colab and draft documentation. The team reviewed
  everything and can explain every part.
- Full source, tests and experiments: https://github.com/nashwu0117/SolarCast-Lite (all rights reserved; see LICENSE)."""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nbf.validate(nb)
    nbf.write(nb, OUT)
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
