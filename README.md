# SolarCast NZ

**English** | [繁體中文](README.zh-TW.md)

Next-hour solar power forecasting for New Zealand, with an interactive map dashboard and a smart EV-charging
decision. Veritas AI International High School Hackathon 2026, **AI + Climate** track.

**Judges: start here** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb)
Runtime > **Run all** (the notebook asks for a free T4 GPU, used only by the AI chat). In about three minutes the
notebook loads our trained model from this repository, shows how accurate it is, and prints a public link (free
Cloudflare tunnel) to the live dashboard; the AI chat is ready a few minutes later.

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
- **Click anywhere in New Zealand** (not only the 14 circles): a blue pin downloads that exact spot's weather from
  Open-Meteo and runs the same model (it borrows the nearest region's climate id; its accuracy is shown for that region).
- **Click** a region: the day's actual vs predicted power, the EV advice for the next hour with its reason,
  and the day's savings (kWh, NZD, kg CO2).
- **Replay** any hour of the test period, or **Live** (New Zealand time; at night solar is zero).
- **Model tab:** our model's accuracy against two baselines, per region, a sample week, and what the model relies on.
- **Ask SolarCast AI** (bottom right): a chat that explains the data in English or Chinese, for example "should the
  Christchurch workshop charge its EV now?". It runs an open model (Qwen3 8B on Colab's T4 GPU, Qwen3 1.7B on a CPU)
  **locally on the notebook's machine**: no internet access, no tools, no API keys. The server looks up the numbers with
  the same code as the map and the model only explains them; a keyword filter, the Llama Guard 3 safety classifier and
  the system rules refuse off-topic or harmful requests; questions are rate-limited. The forecast itself is our own
  model; the chat is an interface on top of it.

## How it works
- **Data:** Open-Meteo hourly weather for 14 New Zealand centres, 2022 to November 2026 (CC BY 4.0).
- **Task:** at the end of hour *t*, predict sunlight (GHI) for hour *t+1*, then convert it to the output of a
  50 kW system. Inputs only use what is known at hour *t*, plus the sun's position (an automated test checks that
  no future data leaks in).
- **Baselines:** persistence (next hour = this hour) and clear-sky persistence (keep this hour's cloudiness).
- **Model (v4):** gradient-boosted trees (scikit-learn), one model for all regions, settings chosen on validation only.
  It averages two models: one predicts sunlight directly, one predicts how cloudy it will be (the clear-sky index, since
  the sun's path is known exactly). Both also read Open-Meteo's **day-ahead forecast** for the hour being predicted
  (issued about 24 hours earlier, so there is no leakage; archived from 2024, earlier rows are missing values).
- **Splits by time:** train 2022-2024, validation Jan-Jun 2025, test Jul 2025-Oct 2026 (reported once).

## Results (test period, daytime hours, all 14 regions)

| Method | Average error (50 kW system) | Error as % of capacity |
|---|---|---|
| Persistence | 4.03 kW | 8.1% |
| Clear-sky persistence | 1.66 kW | 3.3% |
| SolarCast v1 (weather up to now only) | 1.19 kW | 2.4% |
| **SolarCast v4 (current)** | **1.12 kW** | **2.2%** |

SolarCast v4 beats both baselines, and v1, in every region. The learning curve (measured for v1) shows the model needs
about six months of history to beat clear-sky persistence and improves little after one year. Full tables:
`models/nz_model.json`, `experiments/results/nz_learning_curve.md`.

**How v4 was chosen (validation period only, bar fixed in advance: better than v1 by more than 1 W/m2):**
v2 more training sites: no gain; v3 day-ahead forecast inputs: small gain (0.71 W/m2), not enough; v4 forecast inputs
plus the clear-sky-index model averaged with the raw model: 28.98 vs 30.35 W/m2 (gain 1.37), promoted. Details:
`experiments/results/nz_v2.json`, `nz_v3.json`, `nz_v4.json`. The test period was looked at once, for v4 only.

## Limitations (please read)
- **Modelled weather, not real plants.** New Zealand has no public hourly output data for individual solar sites, so
  we use Open-Meteo's modelled (reanalysis) sunlight and derive PV output with a stated formula. Modelled weather is
  smoother than reality, so a real plant would be harder to forecast.
- **Live mode** uses Open-Meteo's forecast service (and its day-ahead forecast of the next hour), a slightly different
  source than the training data, so live accuracy may be a little lower.
- **The workshop and EV are assumptions** (load, charger, prices, CO2 factor), stated in the code.
- **The AI chat can still be wrong.** A small language model can misread the facts, and no safety filter is perfect.
  Without a GPU it is slow, and if the model cannot load the chat answers with fixed templates.

## Repository
```
src/solarcast/nz/   config, data, features, ensemble, model, service, webapp, app, assistant, llm_setup
notebooks/          Judges and Train notebooks, English and 繁體中文
models/             trained model (nz_model.joblib) and its report (nz_model.json)
data/nz/            compact replay data for the dashboard (Open-Meteo, CC BY 4.0)
scripts/            train_nz.py, build_replay_data.py, make_nz_notebooks.py
tests/nz/           pytest suite (offline)
```
Team workflow in Colab: [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md).

## Credits
Weather data: Open-Meteo (CC BY 4.0). Libraries: scikit-learn, pandas, NumPy, Flask. Map: Leaflet and OpenStreetMap
contributors. Charts: Chart.js. Public link: Cloudflare `cloudflared`. Chat: Qwen3 (Apache-2.0) and Llama Guard 3
(Llama 3.2 Community License; Built with Llama), run with Ollama. AI tools: Claude Code helped write code, run
experiments on Colab and draft documentation; the team reviewed everything and can explain every part.
Full list: [NOTICE.md](NOTICE.md).

This project is a prototype and is not energy or financial advice.

## License
All rights reserved; viewing and running for hackathon evaluation is permitted. See [LICENSE](LICENSE). Third-party
data and libraries keep their own licenses; see [NOTICE.md](NOTICE.md).
