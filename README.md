# SolarCast NZ

**English** | [繁體中文](README.zh-TW.md)

How much power will your solar panels give in the next hour, anywhere in New Zealand? SolarCast NZ forecasts it with
machine learning and shows it on an interactive map for any panel size (a 5 kW home or a 50 kW business), with smart EV
charging as a worked example of using the forecast. Veritas AI International High School Hackathon 2026,
**AI + Climate** track.

**Website:** [https://nashwu0117.github.io/SolarCast-Lite/](https://nashwu0117.github.io/SolarCast-Lite/) &middot; **License:** MIT (open source)

**Judges: start here** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb)
Runtime > **Run all** (the notebook asks for a free T4 GPU, used only by the AI chat). In about three minutes the
notebook loads our trained model from this repository, shows how accurate it is, and prints a public link (free
Cloudflare tunnel) to the live dashboard; the AI chat is ready a few minutes later.

| Notebook | Purpose | English | 繁體中文 |
|---|---|---|---|
| Judges | Load the trained model, show results, open the dashboard | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.zh-TW.ipynb) |
| Train | Fetch data, train and evaluate the model, back it up to GitHub | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.zh-TW.ipynb) |

## The user and the decision
Anyone with rooftop solar in New Zealand, from a family with 5 kW of panels to a school or business with 50 kW, gets the
most from their panels by running big loads (an EV charger, a hot-water cylinder, a heat pump, a dishwasher, a
workshop machine) in the hours when the sun is actually there. Clouds make that hard to guess. SolarCast answers one
question for any spot in the country: **how many kW will my panels give next hour?**

**One-sentence pitch:** *SolarCast tells anyone in New Zealand how much power their solar panels will give next hour,
so they use their own sunshine instead of buying from the grid.*

**Worked example (measured):** a workshop with 50 kW of panels and an EV that needs about 40 kWh every working day.
Letting the forecast decide when to charge cuts the grid power the EV uses (see "Does the forecast improve the
decision?" below).

## The dashboard
- **Your panel size** (3, 5, 10, 50 or 100 kW): every kW and kWh on the page is shown for your system. Output scales
  linearly with panel size; the model forecasts sunlight, and power is derived for the chosen size.
- **Map of New Zealand** with 14 regions, coloured by the power your panels would give next hour.
- **Nationwide grid:** the same model runs for 137 land cells (0.5 degrees, about 50 km) covering the whole country, shown
  as a heat map. It is computed before the page opens and recomputed every hour at :50, so the next hour is ready before
  it starts; Live mode refreshes by itself.
- **AI national briefing** above the map: two or three sentences written by the local AI model from the grid numbers
  only (where it is sunniest, North vs South Island, what a 5 kW home system would give), with a fixed template until the
  model is ready.
- **Hover** a region: current weather and whether (and how much) solar power it is generating.
- **Click anywhere in New Zealand** (not only the 14 circles): a blue pin downloads that exact spot's weather from
  Open-Meteo and runs the same model (it borrows the nearest region's climate id; its accuracy is shown for that region).
- **Click** a region: your panels' power next hour, the day's forecast in kWh and what it is worth at the grid price,
  the day's actual vs predicted power, and the worked example (EV advice for the next hour with its reason and the day's
  savings in kWh, NZD, kg CO2).
- **Replay** any hour of the test period, or **Live** (New Zealand time; at night solar is zero).
- **Model tab:** our model's accuracy against two baselines, per region, a sample week, and what the model relies on.
- **Ask SolarCast AI** (bottom right): a chat that explains the data in English or Chinese for your panel size, for example "how much
  will my panels make next hour in Christchurch?". It runs an open model (Qwen3 8B on Colab's T4 GPU, Qwen3 1.7B on a CPU)
  **locally on the notebook's machine**: no internet access, no tools, no API keys. The server looks up the numbers with
  the same code as the map and the model only explains them; a keyword filter, the Llama Guard 3 safety classifier and
  the system rules refuse off-topic or harmful requests; questions are rate-limited. The forecast itself is our own
  model; the chat is an interface on top of it.

## How it works
- **Data:** Open-Meteo hourly weather for 14 New Zealand centres, 2022 to November 2026 (CC BY 4.0).
- **Task:** at the end of hour *t*, predict sunlight (GHI) for hour *t+1*, then convert it to the output of a
  50 kW reference system (any other size is a linear rescale; errors are also reported as % of capacity). Inputs only use what is known at hour *t*, plus the sun's position (an automated test checks that
  no future data leaks in).
- **Baselines:** persistence (next hour = this hour) and clear-sky persistence (keep this hour's cloudiness).
- **Model (v4):** gradient-boosted trees (scikit-learn), one model for all regions, settings chosen on validation only.
  It averages two models: one predicts sunlight directly, one predicts how cloudy it will be (the clear-sky index, since
  the sun's path is known exactly). Both also read Open-Meteo's **day-ahead forecast** for the hour being predicted
  (issued about 24 hours earlier, so there is no leakage; archived from 2024, earlier rows are missing values).
- **Splits by time:** train 2022-2024, validation Jan-Jun 2025, test Jul 2025-Oct 2026 (reported once).

## Results (test period, daytime hours, all 14 regions)

| Method | Average error (50 kW system) | % of capacity | % of mean daytime output (14.3 kW) |
|---|---|---|---|
| Persistence | 4.03 kW | 8.1% | 28.3% |
| Open-Meteo day-ahead forecast used directly | 1.88 kW | 3.8% | 13.2% |
| Clear-sky persistence | 1.66 kW | 3.3% | 11.7% |
| SolarCast v1 (weather up to now only) | 1.19 kW | 2.4% | |
| **SolarCast v4 (current)** | **1.12 kW** | **2.2%** | **7.9%** |

**How to read the accuracy.** "97.8% accuracy" means 100% minus the error as a share of the 50 kW capacity. Measured
against the average daytime output instead, the error is 7.9%. The fairest single number: v4's error is **32.6% lower
than clear-sky persistence** (the strongest simple baseline), 40.3% lower than Open-Meteo's own day-ahead forecast and
72.2% lower than persistence. Errors are larger under cloud: 15.8 W/m2 when cloud cover is at most 60%, 41.1 W/m2 above.

SolarCast v4 beats both baselines, and v1, in every region (14/14). The learning curve (measured for v1) shows the model needs
about six months of history to beat clear-sky persistence and improves little after one year. Full tables:
`models/nz_model.json`, `experiments/results/nz_learning_curve.md`.

**How v4 was chosen (validation period only, bar fixed in advance: better than v1 by more than 1 W/m2):**
v2 more training sites: no gain; v3 day-ahead forecast inputs: small gain (0.71 W/m2), not enough; v4 forecast inputs
plus the clear-sky-index model averaged with the raw model: 28.98 vs 30.35 W/m2 (gain 1.37), promoted. Details:
`experiments/results/nz_v2.json`, `nz_v3.json`, `nz_v4.json`. The test period was looked at once, for v4 only.

**Tried after v4 (v5, validation only, same 1 W/m2 bar against v4's 28.98):** inputs that tell the model how wrong the
day-ahead forecast is right now (forecast vs measured sunlight in the hour that just ended), the forecast's own trend
around the target hour, and five tuning variants. Best: 28.71 W/m2 (bigger trees), a gain of only 0.26, so **v4 was
kept**. The model already gets most of this from the current sunlight plus the target-hour forecast. The test period was
not read for v5. Details: `scripts/train_nz_v5.py`, `experiments/results/nz_v5.json`.

## Run it on your own computer
Python 3.10 or newer:
```bash
git clone https://github.com/nashwu0117/SolarCast-Lite.git
cd SolarCast-Lite
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
python -m solarcast.nz                                 # open http://127.0.0.1:8000
```
Options: `--ai` uses a local [Ollama](https://ollama.com) model for the chat and briefing (first run
`ollama pull qwen3:1.7b`, or `--model qwen3:8b` on a strong computer); `--tunnel` also prints a public Cloudflare link
(needs `cloudflared`); `--no-grid` skips the nationwide grid; `--port 8000`. If the model file does not load, install the
scikit-learn version named in `models/nz_model.json`. Tests: `pip install -e ".[dev]"` then `pytest tests/nz`.

## Does the forecast improve the decision?
The same charging rule (`service.plan_charging`) was driven by different forecasts on every region-day of the test
period (6,818 days) and settled with the actual power. Grid energy caused by the EV, kWh per day (lower is better):

| Strategy | Grid kWh/day | Saved vs charging on arrival |
|---|---|---|
| Charge on arrival (08:00) | 18.01 | - |
| Always charge 10:00-14:00 (a good habit) | 8.46 | 9.55 |
| Persistence forecast | 7.77 | 10.24 |
| Clear-sky persistence forecast | 7.60 | 10.41 |
| **SolarCast v4 forecast** | **6.96** | **11.05** |
| Perfect forecast (upper bound) | 5.31 | 12.70 |

Against the strong 10:00-14:00 habit, SolarCast saves 1.5 kWh/day (about NZD 0.45 and 0.15 kg CO2 per day per
workshop); a perfect forecast would save only 1.65 kWh/day more. The dashboard shows today's savings against both
charging on arrival and the 10:00-14:00 habit. Reproduce: `python experiments/nz_decision_value.py`
(results in `experiments/results/nz_decision_value.md`).

## Limitations (please read)
- **Modelled weather, not real plants.** New Zealand has no public hourly output data for individual solar sites, so
  we use Open-Meteo's modelled (reanalysis) sunlight and derive PV output with a stated formula. Modelled weather is
  smoother than reality, so a real plant would be harder to forecast.
- **Live mode** uses Open-Meteo's forecast service (and its day-ahead forecast of the next hour), a slightly different
  source than the training data, so live accuracy may be a little lower.
- **The workshop and EV are assumptions** (load, charger, prices, CO2 factor), stated in the code.
- **The nationwide grid** borrows the nearest region's climate id for each cell, and its accuracy was measured only
  at the 14 regions; each refresh uses about 270 free Open-Meteo calls, so other replay days are limited per hour.
- **The AI chat can still be wrong.** A small language model can misread the facts, and no safety filter is perfect.
  Without a GPU it is slow, and if the model cannot load the chat answers with fixed templates.

## Repository
```
src/solarcast/nz/   config, data, features, ensemble, model, service, grid, webapp, app, assistant, llm_setup, __main__
notebooks/          Judges and Train notebooks, English and 繁體中文
models/             trained model (nz_model.joblib) and its report (nz_model.json)
data/nz/            compact replay data for the dashboard (Open-Meteo, CC BY 4.0)
data/nz/grid_cells.json  the 137 land cells of the nationwide grid
docs/index.html     the project website (GitHub Pages)
scripts/            train_nz.py, build_replay_data.py, build_grid_cells.py, make_nz_notebooks.py
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
[MIT License](LICENSE): you may use, copy, modify and share the code, notebooks, docs and trained model, keeping the
copyright notice. Third-party data, models and libraries keep their own licenses (for example Open-Meteo data is
CC BY 4.0 and needs attribution); see [NOTICE.md](NOTICE.md).
