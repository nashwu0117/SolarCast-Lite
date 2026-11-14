# SolarCast NZ: module contract

The submission is **New Zealand only**. Everything judges see (notebook, model, web app) uses New Zealand data.
Modules live in `src/solarcast/nz/`. Shared helpers from the older Romania work (`evaluate.regression_metrics`,
`charging.simulate_day`) may be reused; nothing Romanian may appear in NZ outputs.

## Data source

Open-Meteo (CC BY 4.0, free, no key):
- History: `https://archive-api.open-meteo.com/v1/archive` (ERA5-based reanalysis, ~5 days behind real time).
- Live: `https://api.open-meteo.com/v1/forecast` (`past_days` for recent hours, plus the forecast).
- Day-ahead forecasts (model v3/v4 input): `https://previous-runs-api.open-meteo.com/v1/forecast` with the `*_previous_day1`
  variables (`shortwave_radiation`, `cloud_cover`, `temperature_2m`, `direct_normal_irradiance`) = the forecast for hour T
  issued about 24 h earlier. Archived from 2024 (earlier rows are NaN); the same endpoint with `past_days`/`forecast_days`
  returns them for the coming hours too, which is what live mode uses.
Hourly variables: `shortwave_radiation` (GHI, W/m2), `direct_normal_irradiance`, `diffuse_radiation`,
`cloud_cover`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `precipitation`, `is_day`.
Request `timezone=UTC`. **Open-Meteo radiation at timestamp T is the mean over the PRECEDING hour [T-1h, T)**;
other variables are instantaneous at T. Code must state and respect this.

Data is honest about what it is: modelled reanalysis weather, not measurements from real solar plants
(New Zealand has no public per-site hourly PV output data). PV power is derived from GHI with a stated formula.

## `nz/config.py`

- `REGIONS`: ordered dict `name -> (lat, lon)`, 14 centres: Whangarei, Auckland, Hamilton, Tauranga, Gisborne,
  Napier, New Plymouth, Palmerston North, Wellington, Nelson, Christchurch, Queenstown, Dunedin, Invercargill.
- `LOCAL_TZ = "Pacific/Auckland"`, `SYSTEM_KW = 50.0` (the workshop's rooftop system), `PERFORMANCE_RATIO = 0.80`,
  `TEMP_COEFF = -0.004` per degC above 25 degC cell temperature.
- `SPLITS` on the target hour, UTC, half-open: train `2022-01-01..2025-01-01`, val `2025-01-01..2025-07-01`,
  test `2025-07-01..2026-11-01` (test covers a full NZ summer).
- `DAYTIME_ELEVATION_MIN = 5.0`, `MODEL_PATH = models/nz_model.joblib`, `MODEL_META = models/nz_model.json`.

## `nz/data.py`

- `fetch_history(start, end, regions=REGIONS) -> pd.DataFrame`: long format, one row per (region, hour);
  columns `region, time_utc` + the variables above + `solar_elevation_deg`, `solar_azimuth_deg` computed with
  a documented NOAA formula (no extra dependency) at the hour's midpoint. Cached under `data/raw/nz/`.
- `fetch_forecast_history(start, end, regions)`: `region, time_utc, fc_ghi_wm2, fc_cloud_pct, fc_temp_c, fc_dni_wm2`
  (cached under `data/raw/nz/`, clipped to `config.FORECAST_START`). `fetch_history(..., forecasts=True)` adds these columns
  to the long frame at their VALID time (NaN where no forecast exists).
- `fetch_live(regions=REGIONS) -> pd.DataFrame`: same columns (incl. `fc_*`), last 48 h plus next 24 h, plus `is_forecast`.
- `pv_power_kw(ghi, temp_c, system_kw=SYSTEM_KW) -> array`: `system_kw * ghi/1000 * PR * (1 + TEMP_COEFF *
  max(0, cell_temp - 25))`, cell temp = air temp + ghi/800*20 (NOCT-style), clipped to `[0, system_kw]`.

## `nz/features.py`

- Per region, row = issue hour t, target = mean GHI of the next hour (t+1) -> also `target_kw` via `pv_power_kw`.
- Features use only data available at the end of hour t: GHI lags (t, t-1, t-2, t-23), rolling means, weather at t,
  changes over the last hour, target-hour sun elevation/azimuth and calendar (deterministic), clear-sky GHI estimate
  for the target hour, region id (one-hot or ordinal) and latitude.
- Forecast features (v4): `fc_ghi_target, fc_cloud_target, fc_temp_target, fc_dni_target, fc_kt_target` = the day-ahead forecast
  stamped at the TARGET hour t+1 (allowed although t+1 is in the future: it was issued ~24 h before t) and the forecast clear-sky
  index. It is the only thing stamped t+1 or later that is read. `V1_FEATURES` is the original list, `FEATURES = V1_FEATURES + FORECAST_FEATURES`.
- `FEATURES: list[str]`, `build_features(long_df) -> pd.DataFrame` indexed by (`region`, `target_time`), with
  `target_ghi`, `target_kw`, and `meta_*` columns (never model inputs). Includes a leakage test like the Romania one.

## `nz/model.py`

- Baselines: `persistence` (next hour GHI = this hour) and `clear_sky_persistence` (keep this hour's clear-sky
  index, scale by the target hour's clear-sky GHI). Both must be reported next to the model.
- `train(feat_train) -> model`: the v4 `ensemble.SolarEnsemble`, two scikit-learn HistGradientBoostingRegressors pooled over
  all regions (raw GHI, and clear-sky index k = GHI / clear-sky GHI), settings = the v1 settings tuned on val only.
  `predict`: average of the raw model and k * clear-sky GHI (k used where clear-sky GHI >= 20 W/m2, else the raw value), 0 when
  the sun is down. `predict(model, feat) -> Series` (GHI, clipped >= 0), `to_kw(...)`.
- `save(model, metrics)` writes `MODEL_PATH` (joblib) and `MODEL_META` (JSON: sklearn version, feature list,
  train/val/test periods, metrics, plus `version`, `changes`, `previous` (v1 metrics), `experiments` (v2/v3/v4 chain), `report`). `load()` returns the model or `None` if loading fails (e.g. version mismatch),
  so callers can retrain.
- Metrics per region and overall on daytime target hours: MAE/RMSE in W/m2 and in kW for the 50 kW system.

## `nz/service.py` (what the web app calls)

- `Service(model, history_long_df)`; `available_times()` -> list of ISO UTC strings (daytime-rich replay hours);
  `default_time()` -> a NZ local midday in summer of the test period with mixed cloud.
- `snapshot(time_utc: str | "live") -> list[dict]`, one dict per region: `name, lat, lon, local_time,
  cloud_cover_pct, temp_c, wind_kmh, ghi_now_wm2, is_day, predicted_kw_next, persistence_kw_next,
  actual_kw_next (None in live mode), generating (bool: predicted_kw_next > 1)`.
- `detail(region, time_utc | "live") -> dict`: the region's local day as hourly lists (`hours, actual_kw,
  predicted_kw, persistence_kw, cloud_cover_pct`), the EV advice for the next hour (`charge_now: bool`,
  `reason: str`), and the day totals from the charging rule: `grid_kwh_saved, money_saved_nzd, co2_saved_kg`
  versus charging on arrival (NZ assumptions: price NZD 0.30/kWh, grid 0.10 kg CO2/kWh, EV 40 kWh/day,
  11 kW charger, 08:00-18:00 local, workshop load 6 kW), and the region's test metrics.

## `nz/webapp.py`

- `create_app(service) -> Flask app`: `GET /` single-page app; `GET /api/times`, `GET /api/snapshot?t=`,
  `GET /api/detail?region=&t=`. Front end: Leaflet (CDN) map of New Zealand; a circle per region coloured by
  predicted next-hour kW; **hover** tooltip with weather and generation; **click** opens a side panel (below on
  narrow screens) with the detail chart (Chart.js via CDN), EV advice and savings. Time control: Replay slider over
  `available_times()` plus a Live button. Footer: data credits and "Prototype; not energy or financial advice."
- `serve(service, port=8000, tunnel=True) -> str`: runs the app in a background thread; when `tunnel`, downloads
  `cloudflared` (linux-amd64) and returns the `https://*.trycloudflare.com` URL (fallback: localhost URL).
