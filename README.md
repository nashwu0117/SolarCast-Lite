# SolarCast Lite

**English** | [繁體中文](README.zh-TW.md)

One-hour-ahead solar PV power forecasting with scikit-learn, plus an interactive replay demo.
Veritas AI High School Hackathon 2026, **AI + Climate** track.

**Judges: start here** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_Lite_Submission.ipynb) - `notebooks/SolarCast_Lite_Submission.ipynb` is self-contained. Runtime > Run all (about 15 s on a free CPU runtime). Tick `RUN_DASHBOARD` in the last cell for the interactive dashboard (public Cloudflare link).

Team notebooks:

| Notebook | Open in Colab |
|---|---|
| main (submission) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/main.ipynb) |
| 01 data exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/01_data_exploration.ipynb) |
| 02 modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/02_modeling.ipynb) |
| 03 demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/03_demo.ipynb) |

## Project goal (read this first)

**User:** the owner of a small workshop with a 50 kW rooftop solar system and an electric vehicle (a car or a delivery
van) that needs about 40 kWh of charge every working day.

**Their decision, every hour:** *"Should I charge the EV now, or wait?"* Charging when the panels produce a surplus
uses free, clean solar power. Charging at the wrong time means buying electricity from the grid.

**What we build:**
1. **SolarCast forecast** - an ML model that predicts the next hour's solar power from recent power, sun position
   and weather, and beats simple baselines.
2. **Smart charging decision** - each hour, the forecast minus the workshop's own load gives the expected solar
   surplus; the EV charges when the surplus is large enough, and must be full by evening.
3. **Proof it matters** - we replay the held-out test period (Mar-Jun 2026) day by day and compare four charging
   strategies: charge on arrival (no forecast), persistence forecast, **SolarCast**, and perfect foresight (upper
   bound). We report grid electricity avoided (kWh), money saved and CO2 avoided.

**One-sentence pitch:** *SolarCast tells a small business when to charge its EV so it runs on its own sunshine
instead of the grid.*

**What is real and what is assumed:** solar power and weather are real measured data from two 50 kW plants in
Romania. The workshop's load profile and the EV's charging need are synthetic assumptions, stated in the notebook.

```
forecast next hour's solar  ->  minus workshop load  ->  expected surplus
                                                          |-- large enough: charge EV now
                                                          `-- too small: wait (but be full by evening)
```

## Status

| Part | State |
|---|---|
| Data pipeline, leakage-safe features, baselines | Done, 30 tests pass on Colab |
| Forecast models A (power + sun) and B (+ weather) | Done; Model B roughly halves persistence error on the test set |
| Experiments: forecast weather, clear-sky index, pooling and tuning | Results in `experiments/results/` (validation only) |
| EV charging decision simulation | Done; results in the submission notebook, section 5 |
| Submission notebook for judges, dashboard via Cloudflare tunnel | Done, verified on Colab |
| One-page write-up and 2-minute demo video | Team (humans) |

## Problem
Solar plants are only as useful as they are predictable. Clouds change output quickly, so grid operators and plant
owners need a good guess of the next hour's power to balance supply, schedule flexible loads and plan storage.
Better short-term forecasts help solar replace more fossil generation. Electric vehicles are one of the biggest
flexible loads: if they charge at the right hour, they soak up solar power that would otherwise be wasted or sold
cheaply, and they avoid buying grid power.

## Approach
- **Task:** at the end of hour `t`, predict `active_power_kw` of hour `t+1` for a 50 kW plant. Inputs only use data up to `t`
  plus the sun position and calendar of the target hour (no leakage; see `docs/CONTRACT.md`).
- **Baselines:** Persistence (next hour = this hour) and Previous-day same hour.
- **Model A:** gradient boosting on power history and solar geometry.
- **Model B:** Model A plus weather at issue time (cloud cover, irradiance, temperature, ...).
- **Evaluation:** time-based splits (train 2025, validation Jan-Feb 2026, test Mar-Jun 2026), daytime hours, MAE / RMSE and
  MAE as a percentage of capacity; error breakdown by sky condition and power level; cross-site check (A to B).
- **Data:** Two-Site Solar PV Inverter & Weather Dataset (Romania), hourly.

## Repo layout
```
src/solarcast/   config, data, features, baselines, models, evaluate, pipeline, demo
notebooks/       main.ipynb (submission) and 01/02/03 sandboxes (regenerate with scripts/make_notebooks.py)
tests/           pytest suite (offline, synthetic data)
docs/            CONTRACT.md (module interfaces), COLAB_GUIDE.md (team workflow)
scripts/         make_notebooks.py
```

## Local setup
```bash
uv venv --python 3.12 && uv pip install -e ".[demo,dev]"
.venv/bin/python -m pytest -q
.venv/bin/jupyter lab
```

## Colab setup
Every notebook starts with a setup cell that clones the repo and puts `src/` on the Python path. For private-repo
access, saving changes and team rules, see [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md). Run the tests in Colab with
`!cd /content/SolarCast-Lite && PYTHONPATH=src python -m pytest -q`.

## Results
TODO: fill in from `main.ipynb` after the final run (test-set MAE / nMAE for Model B, Model A and both baselines;
error by sky condition; cross-site result).

## Limitations
- **Historical replay.** The demo replays held-out past hours; it is not a live forecasting service.
- **Reanalysis weather.** Weather inputs are Open-Meteo reanalysis (what actually happened), not forecasts, so real-world
  skill would be lower once forecast error is included.
- **Single climate region.** Two plants in Romania, about 1.5 years of data; results may not transfer elsewhere.

## Credits
- Dataset: *Two-Site Solar PV Inverter & Weather Dataset (Romania)*, https://github.com/upb-nlp/PV-Generation-Dataset, CC BY 4.0
  (includes Open-Meteo weather data).
- Solar position: pvlib. Libraries: scikit-learn, pandas, matplotlib, Gradio.
- AI tools: Claude Code.

This project is a prototype.

## License
All rights reserved; viewing and running for hackathon evaluation is permitted. See [LICENSE](LICENSE). Third-party data and libraries keep their own licenses; see [NOTICE.md](NOTICE.md).
