# Experiment: archived weather-forecast features for the target hour

Script: `experiments/nwp_forecast.py` (run on Google Colab, CPU, session `exp-nwp`). Validation split only
(target hours 2026-01-01 to 2026-02-28, daytime). The test split was never scored.

## Method
- Data: `archived_forecasts/site_a_and_b_lead1.csv` (Open-Meteo previous-runs archive, CC BY 4.0).
  Columns: `valid_time_utc, temp_c, humidity_pct, cloud_cover_pct, wind_speed_kmh, ghi_wm2, dni_wm2, dhi_wm2`
  (no precipitation). Both sites share the same file (same weather grid cell).
- Join: for target hour T, take the forecast row whose `valid_time_utc = T + 1 h`, then join on `target_time`.
  Alignment check on train data: correlation of forecast GHI with measured GHI is 0.974 for this alignment
  and 0.934 when shifted by one hour, so the alignment is right.
- Models (same `HistGradientBoostingRegressor` from `models.make_model()`, same rows):
  - Persistence
  - Model A (power + solar), Model B (A + weather at issue hour)
  - **Model C-lite** = A + forecast features, **Model C** = B + forecast features
  - Forecast features (9): `fc_temp_c, fc_humidity_pct, fc_cloud_cover_pct, fc_wind_speed_kmh, fc_ghi_wm2,
    fc_dni_wm2, fc_dhi_wm2` plus `fc_ghi_minus_ghi_now`, `fc_cloud_minus_cloud_now` (forecast for the target
    hour minus the weather measured now).

## Leakage reasoning (for the judges)
- Dataset README: file `lead<k>` holds the run issued `k` days ahead of the hour it describes. The fetch script
  documents it as the value predicted `k x 24 h` before the valid time; `valid_time_utc` is the END of the hour
  (same convention as our hourly weather rows).
- Target hour `[T, T+1h)` is forecast by lead 1 about 24 h before `T + 1h`, i.e. around `T - 23 h`.
  Our issue time is `T` (hour `t = T - 1h` just ended). The forecast existed roughly 23 hours before we need it.
- **Lead 0 is not used**: it is the latest run available for that hour, so its issue time is not guaranteed to be
  before our issue time. Lead 1 is the smallest lead that is safely issued before t. (The dataset toolkit also
  uses lead 1 as its operational "nwp" input.)
- Forecast values are inputs only; no measured value from t+1 or later is used.

## Validation results (daytime, n = 498 rows per site, MAE in kW)

Site A
| Model | MAE kW | RMSE kW | nMAE % |
|---|---|---|---|
| Model A | 2.859 | 4.721 | 5.718 |
| Model C-lite (A+fc) | 2.968 | 4.790 | 5.936 |
| Model C (B+fc) | 3.041 | 5.008 | 6.082 |
| Model B | 3.077 | 5.013 | 6.153 |
| Persistence | 4.295 | 6.850 | 8.589 |

Site B
| Model | MAE kW | RMSE kW | nMAE % |
|---|---|---|---|
| Model A | 2.547 | 4.109 | 5.093 |
| Model B | 2.587 | 4.154 | 5.174 |
| Model C-lite (A+fc) | 2.642 | 4.119 | 5.284 |
| Model C (B+fc) | 2.716 | 4.236 | 5.432 |
| Persistence | 3.559 | 5.750 | 7.117 |

Note: Model A/B on Colab differ slightly from earlier local numbers (A 2.940, B 3.035 at Site A), likely a
scikit-learn version difference; persistence matches exactly (4.295, 3.559).

## Conclusion
- Forecast features give a small gain over Model B at Site A (3.077 -> 3.041 MAE, about 1%), but a loss at Site B
  (2.587 -> 2.716). Model C-lite is worse than Model A at both sites. No forecast variant beats Model A.
- Likely reasons: the daytime val set is small (498 rows, two winter months, low sun), so differences of
  0.05-0.1 kW are within noise; the model already has lags and sun position; the forecast has its own errors
  (GHI correlation with measurement 0.97), and 9 extra columns let the trees fit noise with one year of training.
- The hypothesis "the model needs the weather expected for the target hour" is not supported on this validation set.

## Recommendation
- **Do not integrate** forecast features into `src/` for the submission; keep Model A / Model B as they are.
- If the team still wants to show an "operational" story, add it as an optional documented extra, not the default:
  new columns `fc_temp_c, fc_humidity_pct, fc_cloud_cover_pct, fc_wind_speed_kmh, fc_ghi_wm2, fc_dni_wm2,
  fc_dhi_wm2` from **lead 1**, joined on `target_time` (forecast `valid_time_utc = target_time + 1 h`), as
  feature set "C" = B + those 7 columns. Re-test on a larger validation window (or rolling-origin) before trusting it.
- Spring/summer months have more daytime hours and stronger cloud variability; the test period may differ, but we
  cannot check that here without touching the test split.
