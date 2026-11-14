# SolarCast NZ

**English** | [繁體中文](README.zh-TW.md)

Next-hour solar power forecasting for New Zealand, with an interactive map dashboard and a smart EV-charging
decision. Veritas AI International High School Hackathon 2026, **AI + Climate** track.

**Judges: start here** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb)
Runtime > **Run all**. In about two minutes the notebook loads our trained model from this repository, shows how
accurate it is, and prints a public link (free Cloudflare tunnel) to the live dashboard.

| Notebook | Purpose | English | 繁體中文 |
|---|---|---|---|
| Judges | Load the trained model, show results, open the dashboard | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.zh-TW.ipynb) |
| Train | Fetch data, train and evaluate the model, back it up to GitHub | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.zh-TW.ipynb) |

## The user and the decision
A small New Zealand workshop has a 50 kW rooftop solar system and an electric vehicle that needs about 40 kWh every
working day. Every hour the owner asks: *charge the EV now, or wait for more sun?* Charging when the panels produce
a surplus uses free, clean power; charging at the wrong time buys electricity from the grid.

**One-sentence pitch:** *SolarCast tells a New Zealand business when its own sunshine can charge its EV.*

## The dashboard
- **Map of New Zealand** with 14 regions, coloured by the predicted solar power for the next hour.
- **Hover** a region: current weather and whether (and how much) solar power it is generating.
- **Click** a region: the day's actual vs predicted power, the EV advice for the next hour with its reason,
  and the day's savings (kWh, NZD, kg CO2).
- **Replay** any hour of the test period, or **Live** (New Zealand time; at night solar is zero).
- **Model tab:** our model's accuracy against two baselines, per region, a sample week, and what the model relies on.

## How it works
- **Data:** Open-Meteo hourly weather for 14 New Zealand centres, 2022 to November 2026 (CC BY 4.0).
- **Task:** at the end of hour *t*, predict sunlight (GHI) for hour *t+1*, then convert it to the output of a
  50 kW system. Inputs only use what is known at hour *t*, plus the sun's position (an automated test checks that
  no future data leaks in).
- **Baselines:** persistence (next hour = this hour) and clear-sky persistence (keep this hour's cloudiness).
- **Model:** gradient-boosted trees (scikit-learn), one model for all regions, settings chosen on validation only.
- **Splits by time:** train 2022-2024, validation Jan-Jun 2025, test Jul 2025-Oct 2026 (reported once).

## Results (test period, daytime hours, all 14 regions)

| Method | Average error (50 kW system) | Error as % of capacity |
|---|---|---|
| Persistence | 4.03 kW | 8.1% |
| Clear-sky persistence | 1.66 kW | 3.3% |
| **SolarCast model** | **1.19 kW** | **2.4%** |

SolarCast beats both baselines in every region. The learning curve shows the model needs about six months of history
to beat clear-sky persistence and improves little after one year. Full tables: `models/nz_model.json`,
`experiments/results/nz_learning_curve.md`.

## Limitations (please read)
- **Modelled weather, not real plants.** New Zealand has no public hourly output data for individual solar sites, so
  we use Open-Meteo's modelled (reanalysis) sunlight and derive PV output with a stated formula. Modelled weather is
  smoother than reality, so a real plant would be harder to forecast.
- **Live mode** uses Open-Meteo's forecast service, a slightly different source than the training data, so live
  accuracy may be a little lower.
- **The workshop and EV are assumptions** (load, charger, prices, CO2 factor), stated in the code.

## Repository
```
src/solarcast/nz/   config, data, features, model, service, webapp, app
notebooks/          Judges and Train notebooks, English and 繁體中文
models/             trained model (nz_model.joblib) and its report (nz_model.json)
data/nz/            compact replay data for the dashboard (Open-Meteo, CC BY 4.0)
scripts/            train_nz.py, build_replay_data.py, make_nz_notebooks.py
tests/nz/           pytest suite (offline)
```
Team workflow in Colab: [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md).

## Credits
Weather data: Open-Meteo (CC BY 4.0). Libraries: scikit-learn, pandas, NumPy, Flask. Map: Leaflet and OpenStreetMap
contributors. Charts: Chart.js. Public link: Cloudflare `cloudflared`. AI tools: Claude Code helped write code, run
experiments on Colab and draft documentation; the team reviewed everything and can explain every part.
Full list: [NOTICE.md](NOTICE.md).

This project is a prototype and is not energy or financial advice.

## License
All rights reserved; viewing and running for hackathon evaluation is permitted. See [LICENSE](LICENSE). Third-party
data and libraries keep their own licenses; see [NOTICE.md](NOTICE.md).
