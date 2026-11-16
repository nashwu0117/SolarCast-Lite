# How the SolarCast NZ model works


This page explains, step by step, what data the model uses, how it is trained, and how we measured its accuracy
honestly. Every number here comes from a run on Google Colab; the code that produced it is linked.

## 1. The question the model answers

> Standing at the end of hour *t* in a New Zealand town, how much sunlight will reach the ground during the **next**
> hour (*t+1*), and so how much power will a 50 kW rooftop solar system produce?

The model predicts sunlight (GHI, global horizontal irradiance, in W/m2). We then convert it to power with a stated
formula (section 4). It forecasts one hour ahead, for 14 regions: Whangarei, Auckland, Hamilton, Tauranga, Gisborne,
Napier, New Plymouth, Palmerston North, Wellington, Nelson, Christchurch, Queenstown, Dunedin, Invercargill.

## 2. The data

| Source | What | Period | License |
|---|---|---|---|
| Open-Meteo Historical Weather API (ERA5 reanalysis) | Hourly sunlight (GHI, direct DNI, diffuse DHI), cloud cover, temperature, humidity, wind, rain | 2022-01-01 to 2026-11-08 | CC BY 4.0 |
| Open-Meteo Previous Runs API (`*_previous_day1`) | The weather **forecast** for each hour, as issued about 24 hours earlier (sunlight, cloud, temperature, DNI) | 2024-01-01 onward (earlier forecasts are not archived) | CC BY 4.0 |
| Astronomy (NOAA formulas, computed by us) | Sun elevation and direction for every hour and place | all | - |

**Honest limitation.** New Zealand has no public hourly output data for individual solar plants, so the "truth" we
train and test against is Open-Meteo's modelled (reanalysis) sunlight, not a meter on a real roof. Modelled weather is
smoother than reality, so a real plant would be harder to forecast and the errors below would be larger.

**Timing rule.** Open-Meteo sunlight stamped *T* is the average over the hour *before* *T*. We respect this everywhere:
the target is the value stamped *t+1* (the average over the next hour).

## 3. The information the model gets (features)

All inputs are known at the end of hour *t*. An automated test (`tests/nz/test_nz_features.py`) changes every value
from hour *t+1* onward and checks that no input changes (except the forecast and astronomy for *t+1*, which are known
in advance).

| Group | Features | Why it helps |
|---|---|---|
| Recent sunlight | sunlight now, 1, 2 and 23 hours ago; 3-hour and 24-hour averages; change over the last hour | Weather persists: the next hour often looks like this one |
| Cloudiness | clear-sky index now (sunlight / clear-sky sunlight) and over 24 h; cloud cover now and its change | Separates "dark because cloudy" from "dark because the sun is low" |
| Other weather now | direct and diffuse sunlight, temperature, humidity (and their changes), wind, rain | Hints that clouds are forming or clearing |
| Sun for the next hour | sun elevation and direction, clear-sky sunlight, hour of day, day of year | The biggest driver; astronomy knows it exactly |
| Place | region id, latitude | Each place has its own climate |
| Forecast for the next hour (v4) | forecast sunlight, cloud, temperature, DNI, and forecast clear-sky index, issued ~24 h earlier | Tells the model about clouds that are on the way |

Code: `src/solarcast/nz/features.py`.

## 4. From sunlight to power

`power_kW = 50 * GHI/1000 * 0.80 * (1 - 0.004 * max(0, cell_temp - 25))`, clipped to 0..50 kW, where
`cell_temp = air_temp + GHI/800 * 20`. 0.80 is a typical performance ratio (wiring, inverter, dirt); hot panels lose
0.4% per degree. Code: `src/solarcast/nz/data.py`.

## 5. The model

- **Algorithm:** gradient-boosted decision trees (scikit-learn `HistGradientBoostingRegressor`): hundreds of small
  decision trees, each correcting the previous ones' mistakes. Fast on a free CPU, handles missing values, no GPU.
- **One model for all regions**, trained on daytime hours (the sun is up).
- **v4 (current) is two models averaged:**
  1. one predicts sunlight directly;
  2. one predicts only the clear-sky index (how cloudy the next hour will be), which is then multiplied by the
     clear-sky sunlight that astronomy gives. This lets it focus on clouds.
  Their average is the forecast. Settings: learning rate 0.03, 800 trees, 63 leaves, at least 100 samples per leaf.
- The file is small (a few MB) and lives in `models/` with a JSON report of every metric.

## 6. How we measured accuracy (and kept it honest)

1. **Split by time, never at random.** Train: 2022-2024. Validation: Jan-Jun 2025. Test: Jul 2025-Oct 2026 (a full
   New Zealand summer). Shuffling hours would let the model peek at the future.
2. **Validation is for choices, test is for the final report.** Every design choice (settings, features, v1-v4) was
   made on validation only. The test period is looked at once per promoted model.
3. **A promotion bar fixed in advance:** a new version replaces the old one only if it beats it on validation by
   more than 1 W/m2. Small, possibly lucky gains do not count.
4. **Two baselines** the model must beat: *persistence* (next hour = this hour) and *clear-sky persistence* (keep this
   hour's cloudiness, move the sun forward), which is much harder to beat.
5. **Metric:** mean absolute error (MAE) over daytime hours (sun more than 5 degrees up), in W/m2 and in kW for the
   50 kW system. *Accuracy* = 100% - MAE as a percentage of 50 kW. We also report RMSE and every region separately.
6. **Learning curve:** how the error changes with 7 days to 3 years of training data.

Code: `scripts/train_nz.py`, `src/solarcast/nz/model.py`. Full numbers: `models/nz_model.json`.

## 7. Results

### What we tried (validation, Jan-Jun 2025, daytime MAE, all 14 regions)

| Version | Idea | Validation MAE | Decision |
|---|---|---|---|
| Persistence | next hour = this hour | 104.67 W/m2 | baseline |
| Clear-sky persistence | keep this hour's cloudiness | 41.48 W/m2 | baseline |
| **v1** | gradient boosting on recent weather + sun | 30.35 W/m2 (1.137 kW) | first model |
| v2 | + 16 more training sites (2x data) | 30.04 W/m2 | not promoted (gain 0.31) |
| v3 | + day-ahead weather forecast | 29.64 W/m2 | not promoted (gain 0.71) |
| **v4** | + clear-sky-index model, averaged with v3 | **28.98 W/m2 (1.086 kW)** | **promoted** (gain 1.37) |

**What we learned:** more rows of the same data stopped helping after about a year (learning curve, v2). New
*information* helped (forecasts, v3), and so did letting a model focus on clouds while astronomy handles the sun (v4).

### Learning curve (v1 setup, validation)

| Days of history | 7 | 30 | 90 | 180 | 365 | 730 | all (~1,096) |
|---|---|---|---|---|---|---|---|
| MAE (W/m2) | 48.5 | 44.6 | 37.3 | 31.9 | 30.9 | 30.5 | 30.4 |

### Test period (Jul 2025-Oct 2026, looked at once)

| Method | MAE | Accuracy (% of 50 kW) |
|---|---|---|
| Persistence | 105.4 W/m2 (4.03 kW) | 91.9% |
| Clear-sky persistence | 43.8 W/m2 (1.66 kW) | 96.7% |
| v1 | 31.3 W/m2 (1.19 kW) | 97.6% |
| **v4 (current)** | **29.6 W/m2 (1.12 kW)** | **97.8%** |

v4 beats v1 and both baselines in every one of the 14 regions (best: Christchurch 0.96 kW, 98.1%; hardest:
Whangarei 1.27 kW and Invercargill 1.25 kW, 97.5%). On validation v4 scored 28.98 W/m2 and on test 29.55 W/m2,
so the validation result held up on data the model never saw.

## 8. Where it fails

- **Fast-changing cloud:** the largest errors are hours when clouds arrive or clear within the hour; nothing known at
  hour *t* can fully predict that.
- **Dawn and dusk:** the sun is low, so small timing differences matter.
- **Southern and cloudier regions** (Invercargill, Queenstown) are harder than sunny ones (Christchurch, Napier).
- **Real plants:** modelled weather is smoother than reality (section 2), so real errors will be larger.
