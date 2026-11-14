# SolarCast Lite

One-hour-ahead solar PV power forecasting with scikit-learn, plus an interactive replay demo.
Veritas AI High School Hackathon 2026, **AI + Climate** track.

| Notebook | Open in Colab |
|---|---|
| main (submission) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/main.ipynb) |
| 01 data exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/01_data_exploration.ipynb) |
| 02 modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/02_modeling.ipynb) |
| 03 demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/03_demo.ipynb) |

## Problem
Solar plants are only as useful as they are predictable. Clouds change output quickly, so grid operators and plant
owners need a good guess of the next hour's power to balance supply, schedule flexible loads and plan storage.
Better short-term forecasts help solar replace more fossil generation.

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
Every notebook starts with a setup cell that clones the repo and enables autoreload. For private-repo access, saving
changes and team rules, see [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md).

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
