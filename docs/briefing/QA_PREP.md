# SolarCast NZ: finalist Q&A prep

**English** | [繁體中文](QA_PREP.zh-TW.md)

Format: 3 minutes to present (deck: `SolarCast_NZ_Finals_3min.pptx`, speaker notes inside), then 2 minutes of Q&A.
Every number below comes from `README.md`, `docs/MODEL.md` or `experiments/results/`.

## The 30-second version
SolarCast NZ forecasts how much power solar panels will give in the **next hour**, anywhere in New Zealand, for any
panel size (3 to 100 kW). It uses gradient-boosted trees on free Open-Meteo weather. On data it never saw, the average
error is 1.12 kW on a 50 kW system (2.2% of capacity), 32.6% lower than the strongest simple baseline, and it wins in
14 of 14 regions. The forecast drives an EV charging decision that cuts grid use from 8.46 to 6.96 kWh/day against a
good 10:00-14:00 habit.

## Numbers to remember
| Item | Value |
|---|---|
| Task | Predict sunlight (GHI) for hour t+1, convert to kW |
| Regions / grid | 14 regions; 137 land cells (0.5 degrees) nationwide |
| Data | Open-Meteo, 2022 to Nov 2026, CC BY 4.0 |
| Split | Train 2022-2024; validation Jan-Jun 2025; test Jul 2025-Oct 2026 |
| v4 test error | 29.6 W/m2 = 1.12 kW on 50 kW = 2.2% of capacity = 7.9% of mean daytime output (14.3 kW) |
| vs clear-sky persistence | 32.6% lower error (1.66 kW baseline) |
| vs Open-Meteo day-ahead | 40.3% lower (1.88 kW) |
| vs persistence | 72.2% lower (4.03 kW) |
| Regions beaten | 14 / 14 (best Christchurch 0.96 kW; hardest Whangarei 1.27, Invercargill 1.25) |
| EV test | 6,818 region-days: arrival 18.01, habit 8.46, SolarCast 6.96, perfect 5.31 kWh/day |
| Saving vs habit | 1.5 kWh/day, about NZD 0.45 and 0.15 kg CO2 per day per workshop |
| Horizon | Beats weather service by 40% at 1 h, 16% at 2 h, 6% at 3 h, 0% from 6 h |

The Model tab on the dashboard shows 29.5 W/m2 (29.55 rounded down); the docs say 29.6. Same number.

## Likely questions and short answers

**1. What exactly does it predict?**
Sunlight (GHI, W/m2) for the next hour, then power for a chosen panel size with a stated formula:
`kW = size * GHI/1000 * 0.80 * temperature loss`. Output scales linearly with panel size.

**2. Which AI model, and why that one?**
Gradient-boosted decision trees (scikit-learn HistGradientBoostingRegressor). Hundreds of small trees, each fixing the
previous ones' mistakes. It trains in minutes on a CPU, handles missing values, and suits tabular weather data. v4 is two
models averaged: one predicts sunlight directly, one predicts the clear-sky index (how cloudy), which astronomy then turns
into sunlight.

**3. Why not a deep neural network or an LLM?**
Tabular data with a few years of history: trees are the strong, fast default, and we did not test a neural network
forecaster (be honest if asked). The chat uses an LLM (Qwen3) only to explain numbers; the forecast is our own model.
We compared Qwen3 8B and 14B for the chat only (`experiments/results/nz_llm_compare.md`: both passed 12/12 checks).

**4. How do you know it is not cheating or leaking the future?**
Split by time, never at random. Features use only data known at the end of hour t, plus the sun's position and the
weather service's forecast issued about 24 hours earlier. An automated test (`tests/nz/test_nz_features.py`) changes
every value from t+1 onward and checks no input changes. The test period was looked at once.

**5. 97.8% accuracy sounds too good. What does it mean?**
It is 100% minus the error as a share of the 50 kW capacity. Against mean daytime output the error is 7.9%. We also give
the fairest number: 32.6% lower error than clear-sky persistence.

**6. What are the baselines and why are they fair?**
Persistence (next hour equals this hour) and clear-sky persistence (keep this hour's cloudiness, move the sun forward).
The second is much harder to beat. We also compared against Open-Meteo's own day-ahead forecast.

**7. Is the data real?**
It is Open-Meteo reanalysis (modelled) weather, because New Zealand has no public hourly output for individual solar
plants. Modelled weather is smoother than reality, so a real plant would show larger errors. This is our main
limitation and we say so on the slides.

**8. How did you choose v4? Did you tune on the test set?**
No. All choices (v1 to v4) used validation only, with a bar fixed in advance: a new version must beat the old one by more
than 1 W/m2. v2 (more sites) and v3 (forecast inputs) did not pass; v4 did (gain 1.37). v5 and v6 (extra inputs, XGBoost
on GPU) did not pass either, so we kept v4. The test period was read once.

**9. Why only one hour ahead?**
Counted from the latest weather data, we beat the weather service by 40% at 1 hour, 16% at 2 hours, 6% at 3 hours and not
at all from 6 hours. We stop where we add value (`experiments/results/nz_horizons.md`).

**10. How much history does it need?**
About three months: with 90 days of training the error (37.3 W/m2) is already below clear-sky persistence (41.5), and gains flatten after about one year (learning curve on v1; the README says six months, which is more conservative).

**11. Where does it fail?**
When clouds arrive or clear inside the hour, at dawn and dusk, and in cloudier southern regions (Invercargill,
Queenstown). Error is 15.8 W/m2 when cloud cover is at most 60%, 41.1 W/m2 above.

**12. How does it work for any location, not just 14 regions?**
Clicking anywhere downloads that spot's weather from Open-Meteo and runs the same model, borrowing the nearest region's
climate id. Accuracy was measured only at the 14 regions. The nationwide grid does the same for 137 cells every hour.

**13. How is the EV result measured, and is it realistic?**
The same charging rule was driven by different forecasts on 6,818 region-days and settled with actual power. The car,
charger, prices (NZD 0.30/kWh) and CO2 factor (0.10 kg/kWh) are stated assumptions in the code, so treat savings as an
illustration of the forecast's value, not a bill prediction.

**14. Why is the saving only 1.5 kWh/day?**
Because the 10:00-14:00 habit is already good. A perfect forecast would save only 1.65 kWh/day more, so v4 captures most of
what is possible. The bigger saving is against charging on arrival (11.05 kWh/day).

**15. What is the AI chat, and is it safe?**
An open model (Qwen3) running locally on the notebook's machine: no internet, no tools, no API keys. The server looks up
the numbers with the same code as the map; the model only explains them. A keyword filter, the Llama Guard 3 classifier
and system rules refuse off-topic or harmful requests, and questions are rate-limited. It can still be wrong, and falls back
to fixed templates if the model cannot load.

**16. Who benefits, and what is the climate impact?**
Anyone with rooftop solar: a 5 kW home or a 50 kW business. Using more of your own sunshine means less grid energy and
fewer emissions. Our measured example is small per site; the value is in being usable by many sites at no cost.

**17. How did you use AI tools?**
Claude Code helped write code, run experiments on Colab and draft documentation. The team reviewed everything and can
explain every part (stated in the README credits).

**18. Is it open source? Can I reproduce it?**
Yes, MIT license. Colab and Kaggle Judges notebooks load our trained model, show the results and open the dashboard in
about three minutes. Training notebooks and experiment scripts are in the repository.

**19. What would you do next?**
Compare with real rooftop output, extend beyond one hour if it adds value, and measure real-world EV savings.

**20. What was the hardest part?**
Keeping the evaluation honest: no future leakage, a fixed promotion bar, and reporting that more data did not help but new
information (forecasts, the clear-sky index) did.

## If something goes wrong in the demo
- Use the screenshots in `docs/briefing/screenshots/` (replay, EV panel, Model tab) or the website
  <https://nashwu0117.github.io/SolarCast-Lite/>.
- Local run: `pip install -e .` then `python -m solarcast.nz` and open <http://127.0.0.1:8000>.
  If the model fails to load, install the scikit-learn version named in `models/nz_model.json` (1.6.1).
- At night the Live view shows 0 kW. Press **Replay today's sun** for a daytime demo.
