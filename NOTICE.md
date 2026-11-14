# Third-party notices and credits

**English** | [繁體中文](NOTICE.zh-TW.md)

SolarCast Lite's own code is covered by [LICENSE](LICENSE) (all rights reserved). The material below belongs to
others and is used under its own license. The hackathon rules require crediting every dataset, model, tutorial and
AI tool; this file is that list.

## Data

| Material | Source | License | How we use it |
|---|---|---|---|
| Two-Site Solar PV Inverter & Weather Dataset (Romania) | https://github.com/upb-nlp/PV-Generation-Dataset | CC BY 4.0 | Downloaded at run time (not redistributed in this repo). Hourly PV power, weather and solar position for Site A and Site B; archived forecasts in an experiment. |
| Open-Meteo weather (historical and archived forecasts) | https://open-meteo.com, included in the dataset above | CC BY 4.0 | Weather features, via the dataset. |
| Solar position | pvlib (https://pvlib-python.readthedocs.io), computed by the dataset authors | BSD-3-Clause | Solar elevation and azimuth columns, via the dataset. |

Attribution required by CC BY 4.0: *"Two-Site Solar PV Inverter & Weather Dataset (Romania)" by the upb-nlp
dataset authors, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). We derived features,
models and evaluation results from it; the original data is unchanged.* See the dataset's own `CITATION.cff` for
the preferred citation.

## Software libraries

| Library | License |
|---|---|
| scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| Matplotlib | Matplotlib License (PSF-based) |
| Gradio | Apache-2.0 |
| pytest, nbformat, nbconvert, Jupyter | MIT / BSD-3-Clause |
| Google Colab CLI (google-colab-cli), used to run everything on Colab | Apache-2.0 |

## AI tools

- **Claude Code** (Anthropic): used to write and review code, run experiments on Colab and draft documentation.
  The team reviewed the work and can explain every part of the project.

## Assumptions (not third-party data)

The workshop load profile, EV charging need, electricity price and grid CO2 intensity in
`src/solarcast/charging.py` are our own synthetic assumptions, not measured data.
