# Third-party notices and credits

**English** | [繁體中文](NOTICE.zh-TW.md)

SolarCast NZ's own code is covered by [LICENSE](LICENSE) (all rights reserved). The material below belongs to
others and is used under its own license. The hackathon rules require crediting every dataset, pretrained model,
tutorial and AI tool; this file is that list.

## Data

| Material | Source | License | How we use it |
|---|---|---|---|
| Open-Meteo Historical Weather API (ERA5 reanalysis) | https://open-meteo.com/en/docs/historical-weather-api | CC BY 4.0 | Hourly weather for 14 New Zealand centres, 2022 to Nov 2026: training, validation, test. |
| Open-Meteo Previous Runs API (archived day-ahead forecasts) | https://open-meteo.com/en/docs/previous-runs-api | CC BY 4.0 | Day-ahead forecast inputs of model v4 (from 2024). |
| Open-Meteo Forecast API | https://open-meteo.com/en/docs | CC BY 4.0 | Live mode of the dashboard. |
| `data/nz/replay.csv.gz` | a subset of the Open-Meteo data above | CC BY 4.0 | Replay data for the dashboard (test period only). |

Attribution required by CC BY 4.0: *Weather data by Open-Meteo.com (https://open-meteo.com), licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). We derived features, PV power estimates, models and evaluation results
from it.*

## Pretrained models (SolarCast AI chat only; the solar forecast is our own model)

| Model | Source | License | How we use it |
|---|---|---|---|
| Qwen3 8B and Qwen3 1.7B (Alibaba Qwen team) | https://huggingface.co/Qwen/Qwen3-8B, https://ollama.com/library/qwen3 | Apache-2.0 | The dashboard chat explains our data in plain language (8B on a GPU, 1.7B on a CPU). Not fine-tuned. |
| Llama Guard 3 1B (Meta) | https://huggingface.co/meta-llama/Llama-Guard-3-1B, https://ollama.com/library/llama-guard3 | Llama 3.2 Community License | Safety classifier that checks chat questions. **Built with Llama.** |

## Software

| Software | License | Use |
|---|---|---|
| scikit-learn | BSD-3-Clause | Our forecasting model (gradient boosting) |
| pandas, NumPy | BSD-3-Clause | Data handling |
| Matplotlib | Matplotlib License (PSF-based) | Notebook charts |
| Flask, Werkzeug | BSD-3-Clause | Dashboard web server |
| Leaflet | BSD-2-Clause | Map |
| OpenStreetMap map tiles | data (c) OpenStreetMap contributors, ODbL | Map background |
| Chart.js | MIT | Dashboard charts |
| Cloudflare cloudflared | Apache-2.0 | Free public link (quick tunnel) |
| Ollama | MIT | Runs the open language models locally on the Colab VM |
| pytest, nbformat, nbconvert, Jupyter | MIT / BSD-3-Clause | Tests and notebook runs |
| Google Colab CLI (google-colab-cli) | Apache-2.0 | Running everything on Colab |

## AI tools

- **Claude Code** (Anthropic): used to write and review code, run experiments on Colab and draft documentation.
  The team reviewed the work and can explain every part of the project.

## Assumptions (not third-party data)

The workshop load, EV charging need, charger power, electricity price and grid CO2 intensity in
`src/solarcast/nz/config.py` are our own stated assumptions, not measured data.
