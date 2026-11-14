# SolarCast Lite: module contract

Every module in `src/solarcast/` follows this contract so the pieces fit together
and teammates can work on them in parallel. Constants live in `config.py`; never
hard-code a path, date or column list anywhere else. Change this file first
if an interface has to change.

## Task definition

Row `t` of a site's hourly CSV covers the UTC hour `[t, t+1h)`; its weather is the
Open-Meteo record valid at the end of that hour. At **issue time** (the moment hour `t`
is complete) we predict **`active_power_kw` of hour `t+1`** (the **target hour**).

Allowed inputs: anything from rows `<= t`, plus the solar position and calendar of the
target hour (deterministic, computable in advance).
Forbidden: power or weather from rows `>= t+1`. `tests/test_features.py` checks this.

## `data.py`

- `download_site_csv(site: str, force: bool = False) -> pathlib.Path`
  Downloads `DATASET_BASE_URL/SITES[site]["csv"]` into `DATA_DIR` (created if needed);
  reuses the cached file unless `force`.
- `load_site(site: str) -> pd.DataFrame`
  - index: `DatetimeIndex`, tz-aware **UTC**, name `"datetime_utc"`, sorted, unique,
    complete hourly grid (reindex to `freq="h"` from first to last row).
  - columns: exactly `config.KEEP_COLS`, float dtype.
  - missing measurements stay `NaN`; never fill them with 0. Night zeros are real zeros.
- `data_quality_report(df) -> dict` with keys `n_rows`, `start`, `end`,
  `n_missing_power`, `n_missing_weather` (rows with any weather NaN),
  `n_duplicate_timestamps_in_csv`, `n_inserted_gap_rows`, `pct_daytime`.

## `features.py`

- `FEATURE_GROUPS: dict[str, list[str]]` with keys `"power"`, `"weather"`, `"time_solar"`.
- `FEATURE_SETS: dict[str, list[str]]`:
  `"A"` (power + solar) = power + time_solar; `"B"` (weather-aware) = A + weather.
- `build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame`
  - input: output of `load_site`.
  - index: the **target hour** (UTC, name `"target_time"`).
  - columns: every feature in `FEATURE_SETS["B"]`, plus `target` (power of the target hour),
    plus metadata columns starting with `meta_` that are **never model inputs**:
    `meta_issue_time`, `meta_target_elevation`, `meta_target_cloud_cover`,
    `meta_target_capacity_factor`.
  - `dropna=True` drops rows where `target` or any `FEATURE_SETS["B"]` column is NaN,
    so models A and B and the baselines are scored on exactly the same rows.

Feature names (issue hour = `t`, target hour = `t+1`):

| group | columns |
|---|---|
| power | `power_lag1` (power at t), `power_lag2`, `power_lag3`, `power_lag6`, `power_roll3_mean` (t-2..t), `power_roll6_mean` (t-5..t), `power_same_hour_yesterday` (power at t+1-24h) |
| weather | the 8 `WEATHER_COLS` at t, `cloud_cover_diff1` (t minus t-1), `ghi_diff1` (t minus t-1) |
| time_solar | `target_hour_sin`, `target_hour_cos` (local hour of t+1), `target_doy_sin`, `target_doy_cos`, `target_solar_elevation`, `target_solar_azimuth`, `issue_solar_elevation` |

## `baselines.py`

Each takes the `build_features` frame and returns a `pd.Series` (same index, kW):
- `persistence(feat)`: `power_lag1`
- `previous_day_same_hour(feat)`: `power_same_hour_yesterday`
- `BASELINES = {"Persistence": persistence, "Previous-day same hour": previous_day_same_hour}`

## `models.py`

- `make_model(random_state=config.RANDOM_STATE)`: an unfitted `HistGradientBoostingRegressor`.
- `train_model(feat_train, feature_set: str)`: fits on `FEATURE_SETS[feature_set]` columns
  and `target`; returns the fitted model with `model.feature_set = feature_set`.
- `predict(model, feat) -> pd.Series`: uses `model.feature_set` columns, clips to `[0, CAPACITY_KW]`.
- `save_model(model, name) -> Path`, `load_model(name)`: joblib under `MODELS_DIR`.
- `permutation_importance_table(model, feat, n_repeats=5) -> pd.DataFrame`
  (columns `feature`, `importance_mean`, `importance_std`, sorted descending).

## `evaluate.py`

- `split_by_time(feat, splits=config.SPLITS) -> dict[str, pd.DataFrame]`, keys `train`/`val`/`test`.
- `daytime_mask(feat) -> pd.Series[bool]`: `meta_target_elevation > DAYTIME_ELEVATION_MIN`.
- `regression_metrics(y_true, y_pred, capacity_kw=CAPACITY_KW) -> dict`:
  `mae_kw`, `rmse_kw`, `nmae_pct` (MAE / capacity × 100), `n`. No MAPE.
- `compare(feat, predictions: dict[str, pd.Series], daytime_only=True) -> pd.DataFrame`:
  one row per model name, metric columns, sorted by `mae_kw`.
- `grouped_metrics(feat, predictions, by: str, daytime_only=True) -> pd.DataFrame`:
  `by="sky"` groups with `config.SKY_BINS` on `meta_target_cloud_cover`;
  `by="power_bin"` groups actual power into 0-10, 10-20, 20-30, 30-40, 40-50 kW.
  Long format: columns `group`, `model`, `mae_kw`, `rmse_kw`, `nmae_pct`, `n`.
- `worst_errors(feat, predictions, model: str, k=10) -> pd.DataFrame`.
- Plot helpers return a matplotlib `Figure` and never call `plt.show()`:
  `plot_actual_vs_predicted(feat, predictions, start=None, end=None)`,
  `plot_error_by_group(grouped_df)`.

## `pipeline.py`

- `run_experiment(site: str = "A") -> ExperimentResult` (a dataclass) with fields
  `site`, `df` (load_site output), `feat`, `splits` (dict), `models` (`{"Model A": ..., "Model B": ...}`),
  `predictions` (`{split_name: {model_name: Series}}` for val and test, including both baselines),
  `metrics` (`{split_name: compare(...) DataFrame}`).
  Models train on `train` only. `val` is for choices. `test` is reported once.
- `cross_site(result_a, result_b) -> pd.DataFrame`: Site A's models scored on Site B's test split.

## `demo.py`

- `predict_at(result, target_time) -> dict`: predicted kW for Model B and Model A, both
  baselines, actual, absolute error, capacity factor, `recent_power` (Series, last 6 h before
  issue), `weather_now` (dict at issue hour).
- `build_demo(results: dict[str, ExperimentResult]) -> gradio.Blocks`: site selector (A/B),
  a time selector limited to daytime **test** hours, recent-power chart, weather summary,
  prediction vs baselines vs actual, actual vs predicted chart for that day, and the test-set
  metrics table. Imports gradio lazily inside the function.

## Notebooks (`notebooks/`)

Each one opens in Colab and runs top to bottom with **Restart and run all**. The first cell
is the shared Colab setup cell (clone or pull the repo, add `src/` to `sys.path`, install gradio if missing, autoreload).
- `main.ipynb`: end-to-end submission notebook (data, baselines, models A/B, test metrics,
  error analysis, demo). Saved with outputs.
- `01_data_exploration.ipynb` (Data Lead), `02_modeling.ipynb` (ML Lead),
  `03_demo.ipynb` (Product & Demo Lead): sandboxes that import from `solarcast`.
