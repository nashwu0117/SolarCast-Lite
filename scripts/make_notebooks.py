"""Regenerate every notebook in notebooks/ (saved without outputs).

Usage: python scripts/make_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP_MD = """## Setup (run this first)
This cell makes the notebook work both in Google Colab and on a laptop. In Colab it clones (or updates) the
GitHub repo into `/content/SolarCast-Lite`; locally it finds the repo root. Then it puts `src/` on the Python path
and turns on autoreload, so edits to `src/solarcast/*.py` are picked up without restarting.
If the repo is private, add a Colab Secret named `GITHUB_TOKEN` (see `docs/COLAB_GUIDE.md`). The token is never printed."""

SETUP_CODE = '''import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/nashwu0117/SolarCast-Lite.git"
IN_COLAB = "google.colab" in sys.modules


def _git(args, cwd=None, secret=None, check=True):
    """Run git, hiding the token if it shows up in any output."""
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if secret:
        out = out.replace(secret, "***")
    if check and r.returncode != 0:
        raise RuntimeError("git " + args[0] + " failed:\\n" + out)
    return r.returncode, out


if IN_COLAB:
    REPO = Path("/content/SolarCast-Lite")
    token = None
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = None  # no secret: fine for a public repo
    auth_url = REPO_URL.replace("https://", f"https://x-access-token:{token}@") if token else REPO_URL
    if not REPO.exists():
        _git(["clone", auth_url, str(REPO)], secret=token)
        _git(["remote", "set-url", "origin", REPO_URL], cwd=REPO)  # keep the token out of .git/config
    else:
        _git(["remote", "set-url", "origin", auth_url], cwd=REPO, secret=token)
        code, out = _git(["pull", "--ff-only"], cwd=REPO, secret=token, check=False)
        _git(["remote", "set-url", "origin", REPO_URL], cwd=REPO)
        if code != 0:
            print("git pull did not fast-forward (local edits?). Continuing with the current files.\\n" + out)
    os.chdir(REPO)
    if importlib.util.find_spec("gradio") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
else:
    REPO = Path.cwd().resolve()
    while not (REPO / "pyproject.toml").exists() and REPO != REPO.parent:
        REPO = REPO.parent
    os.chdir(REPO)

# Editable installs do not take effect in Colab without a runtime restart, so use the path directly.
sys.path.insert(0, str(REPO / "src"))
print("Repo:", REPO, "| Colab:", IN_COLAB)

%load_ext autoreload
%autoreload 2'''

LAUNCH_CODE = '''import os

from solarcast.demo import build_demo

demo = build_demo({"A": res_a, "B": res_b})
# Set SOLARCAST_LAUNCH_DEMO=0 to build the app without launching it (used for automated checks).
if os.environ.get("SOLARCAST_LAUNCH_DEMO", "1") != "0":
    demo.launch(share=IN_COLAB)  # Colab: public link; locally: inline'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.validate(nb)
    nbf.write(nb, OUT / name)
    print("wrote", name)


def head(job):
    return [md(job), md(SETUP_MD), code(SETUP_CODE)]


def main():
    OUT.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ main.ipynb
    save("main.ipynb", [
        md("""# SolarCast Lite: one-hour-ahead solar power forecasting
Veritas AI High School Hackathon 2026, AI + Climate track.

**Problem and user.** A small solar plant (50 kW) feeds the grid, and whoever operates it needs to know how much power
it will produce in the next hour: to balance supply and demand, and to decide when to run flexible loads or storage.
Clouds make the next hour hard to guess. We forecast the power of the next hour from recent power, solar geometry and
weather, with scikit-learn, and compare against two simple baselines.

**Important.** Results are a *historical replay* on held-out test data. The weather inputs are Open-Meteo *reanalysis*
(what the weather actually was), not live forecasts. This project is a prototype."""),
        md(SETUP_MD), code(SETUP_CODE),
        md("## 1. Data and quality report (Site A)\nTwo-Site Solar PV Inverter & Weather Dataset (Romania), hourly. "
           "Missing measurements stay `NaN`; they are never filled with zeros."),
        code("""import pandas as pd

from solarcast import config, data

df_a = data.load_site("A")
print(pd.Series(data.data_quality_report(df_a)))
df_a.head()"""),
        md("""## 2. Features and the no-leakage rule
Row `t` covers the hour ending at `t+1h`. At **issue time** (hour `t` is complete) we predict the power of hour `t+1`.
Inputs may only use rows up to `t`, plus the sun position and calendar of the target hour (known in advance).
Power or weather from `t+1` or later is never used (`tests/test_features.py` checks this).

- **Model A (power + solar):** recent power lags and rolling means, power at the same hour yesterday, sun position and time of day.
- **Model B (weather-aware):** Model A's features plus weather at issue time (cloud cover, irradiance, temperature, ...) and how fast cloud cover and irradiance are changing.
- **Baselines:** *Persistence* (next hour = this hour) and *Previous-day same hour*.
- **Splits (by time, never shuffled):** train 2025, validation Jan-Feb 2026, test Mar-Jun 2026. Models train on `train` only."""),
        code("""from solarcast.features import FEATURE_GROUPS, build_features

for group, cols in FEATURE_GROUPS.items():
    print(f"{group:11s}", cols)
feat_a = build_features(df_a)
print(feat_a.shape)"""),
        md("## 3. Train and evaluate (Site A and Site B)"),
        code("""from solarcast.pipeline import run_experiment

res_a = run_experiment("A")
res_b = run_experiment("B")"""),
        md("### Metrics: validation and test, Site A (daytime hours)\n`nmae_pct` is MAE as a percentage of the 50 kW capacity."),
        code("""print("Validation (used for choices)")
display(res_a.metrics["val"].round(3))
print("Test (reported once)")
display(res_a.metrics["test"].round(3))"""),
        md("### Actual vs predicted, one sample test week"),
        code("""from solarcast.evaluate import plot_actual_vs_predicted

plot_actual_vs_predicted(res_a.splits["test"], res_a.predictions["test"],
                        start="2026-05-18", end="2026-05-24")"""),
        md("## 4. Error analysis (test split, Site A)\nWhere does the model do well, and where does it fail?"),
        code("""from solarcast.evaluate import grouped_metrics, plot_error_by_group

by_sky = grouped_metrics(res_a.splits["test"], res_a.predictions["test"], by="sky")
display(by_sky.round(3))
plot_error_by_group(by_sky)"""),
        code("""by_power = grouped_metrics(res_a.splits["test"], res_a.predictions["test"], by="power_bin")
display(by_power.round(3))
plot_error_by_group(by_power)"""),
        md("### Worst errors (Model B)"),
        code("""from solarcast.evaluate import worst_errors

worst_errors(res_a.splits["test"], res_a.predictions["test"], "Model B", k=10).round(2)"""),
        md("## 5. What does Model B use? (permutation importance)\nHow much the test MAE gets worse when one feature is shuffled."),
        code("""from solarcast.models import permutation_importance_table

imp = permutation_importance_table(res_a.models["Model B"], res_a.splits["test"])
display(imp.head(12).round(4))
imp.head(12).set_index("feature")["importance_mean"].iloc[::-1].plot.barh(
    figsize=(7, 4), title="Permutation importance (MAE increase, kW)");"""),
        md("## 6. Cross-site check\nSite A's trained models scored on Site B's test split (no retraining)."),
        code("""from solarcast.pipeline import cross_site

print("Site B, own models")
display(res_b.metrics["test"].round(3))
print("Site B test split, Site A models")
display(cross_site(res_a, res_b).round(3))"""),
        md("""## 7. Interactive demo
Pick a site and a daytime hour from the test split to see the one-hour-ahead forecast next to the baselines and the
actual value. In Colab this launches with a public `share=True` link.
*Historical replay on held-out test data. Prototype.*"""),
        code(LAUNCH_CODE),
        md("""## Credits
- Dataset: *Two-Site Solar PV Inverter & Weather Dataset (Romania)*, https://github.com/upb-nlp/PV-Generation-Dataset, licensed CC BY 4.0. It includes Open-Meteo weather data.
- Solar position: pvlib (computed in the dataset).
- Libraries: scikit-learn, pandas, numpy, matplotlib, Gradio.
- AI tools: Claude Code was used to help write code and documentation.

This project is a prototype. Results are a historical replay; weather inputs are Open-Meteo reanalysis, not live forecasts."""),
    ])

    # ------------------------------------------------------------ 01_data_exploration
    save("01_data_exploration.ipynb", head("""# 01 Data exploration (Data Lead sandbox)
## Your job in this notebook
Understand the data and the features: look for gaps, outliers and odd patterns, and check that features never use
future information. Your files: `src/solarcast/data.py`, `features.py`, `baselines.py`. Try ideas here, then move
working code into those files. Do not edit this notebook's setup cell.""") + [
        code("""import matplotlib.pyplot as plt
import pandas as pd

from solarcast import config, data

df = data.load_site("A")
print(pd.Series(data.data_quality_report(df)))
df.describe().T.round(2)"""),
        md("### Power on a few sample days"),
        code("""df.loc["2025-06-10":"2025-06-14", config.TARGET_COL].plot(figsize=(10, 3), ylabel="kW");"""),
        md("### Average daily profile by local hour"),
        code("""local = df.tz_convert(config.LOCAL_TZ)
local.groupby(local.index.hour)[config.TARGET_COL].mean().plot(figsize=(7, 3), xlabel="local hour", ylabel="mean kW");"""),
        md("### Does cloud cover explain power? (daytime only)"),
        code("""day = df[df["solar_elevation_deg"] > config.DAYTIME_ELEVATION_MIN]
day.plot.scatter(x="cloud_cover_pct", y=config.TARGET_COL, alpha=0.1, figsize=(6, 4));"""),
        md("### Features (note the `meta_` columns are never model inputs)"),
        code("""from solarcast.features import build_features

feat = build_features(df)
feat.head()"""),
        md("### Your experiments\nAdd your own cells below."),
    ])

    # ----------------------------------------------------------------- 02_modeling
    save("02_modeling.ipynb", head("""# 02 Modeling (ML Lead sandbox)
## Your job in this notebook
Train and compare models on the **train** and **validation** splits: try feature sets, model settings and ideas.
Your files: `src/solarcast/models.py`, `evaluate.py`, `pipeline.py`. Move what works into those files.""") + [
        md("""> **Warning: do not tune on the test split.**
> Use `train` to fit and `val` to choose. The test split (Mar-Jun 2026) is looked at once, at the end, for reporting.
> If you change a setting because it improved test scores, the reported numbers are no longer honest."""),
        code("""from solarcast import models
from solarcast.baselines import BASELINES
from solarcast.evaluate import compare, split_by_time
from solarcast.features import build_features
from solarcast.data import load_site

feat = build_features(load_site("A"))
splits = split_by_time(feat)
{k: v.shape for k, v in splits.items()}"""),
        md("### Train Model A and Model B, score on validation"),
        code("""preds = {name: fn(splits["val"]) for name, fn in BASELINES.items()}
for name, fs in (("Model A", "A"), ("Model B", "B")):
    m = models.train_model(splits["train"], fs)
    preds[name] = models.predict(m, splits["val"])
compare(splits["val"], preds).round(3)"""),
        md("### Your experiments (validation only)\nExample ideas: other model settings in `make_model`, extra features, different lags."),
        code("""# Example: try a different model setting on validation
from sklearn.ensemble import HistGradientBoostingRegressor

# m2 = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, random_state=42)"""),
    ])

    # --------------------------------------------------------------------- 03_demo
    save("03_demo.ipynb", head("""# 03 Demo (Product & Demo Lead sandbox)
## Your job in this notebook
Build and polish the Gradio demo and the product story. Your files: `src/solarcast/demo.py`, this notebook, `main.ipynb`
and `README.md`. Edit `demo.py`, then re-run the cells below (autoreload picks up your changes).""") + [
        code("""from solarcast.pipeline import run_experiment

res_a = run_experiment("A")
res_b = run_experiment("B")"""),
        md("### One prediction, without the UI"),
        code("""from solarcast.demo import predict_at

t = res_a.splits["test"].index[500]
info = predict_at(res_a, t)
{k: v for k, v in info.items() if k not in ("recent_power", "weather_now")}"""),
        md("### Launch the demo\nIn Colab this creates a public `share=True` link. *Historical replay on held-out test data. Prototype.*"),
        code(LAUNCH_CODE),
    ])


if __name__ == "__main__":
    main()
