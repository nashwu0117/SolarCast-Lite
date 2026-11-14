# Experiment: clear-sky normalisation of the target (validation split only)

Script: `experiments/clear_sky.py` (run on Google Colab with `PYTHONPATH=src python experiments/clear_sky.py`).
All numbers are Jan-Feb 2026 (val), daytime rows only (target elevation > 5 deg), n = 498 rows per site.
The test split is never loaded into any model or metric.

## Method

Two estimates of "clear-sky power" (csp, kW), both fitted on the TRAIN split (2025) only:

- **env** (empirical envelope): for every (day of year +- 10 days, local hour of the target) the 95th percentile of train power.
- **sin** (physical proxy): `csp = Q * sin(solar elevation)`, with `Q` = 95th percentile of `power / sin(elevation)` over train daytime rows.

Two ways to use csp, both with the same `HistGradientBoostingRegressor` from `make_model()`:

1. **CS features** (still predicting kW): add `csp_target`, `csp_issue`, `ci_now = power_lag1 / csp_issue`, `ci_roll3 = power_roll3_mean / csp_issue`.
2. **k-index**: train on `k = target / csp_target` (clipped to [0, 1.5], sample weight = csp so the loss matches kW error), then `pred_kW = k_hat * csp_target`.

Safety for small denominators: rows with csp < 2 kW are not used to train the k-model, `ci_*` is NaN where `csp_issue < 2 kW`,
and at prediction time rows with `csp_target < 2 kW` (or no envelope available) fall back to the raw-kW Model B prediction.

## Why there is no leakage

- csp is built only from train rows (envelope quantiles / the scale `Q`); validation power never enters it.
- At prediction time csp needs only the calendar and sun position of the target hour and of the issue hour, which are known in advance (same rule as `target_solar_elevation` in `CONTRACT.md`).
- `ci_now` uses `power_lag1` (hour t) and csp of hour t, so only information available at issue time.
- Hyper-parameters are not tuned on val; the variants are fixed in advance. (Caveat: with 498 rows, picking the "best" of 7 variants on val is itself optimistic.)

## Results (as printed on Colab)

Site A, val daytime:

| model | MAE kW | RMSE kW | nMAE % |
|---|---|---|---|
| Model A | 2.859 | 4.721 | 5.718 |
| A + CS feats (env) | 2.860 | 4.820 | 5.719 |
| B + CS feats (env) | 2.976 | 5.037 | 5.951 |
| B + CS feats k-index (env) | 2.977 | 4.795 | 5.955 |
| B + CS feats (sin) | 2.979 | 4.914 | 5.958 |
| B k-index (env) | 3.040 | 4.931 | 6.080 |
| B k-index (sin) | 3.047 | 4.932 | 6.095 |
| Model B | 3.077 | 5.013 | 6.153 |
| Persistence | 4.295 | 6.850 | 8.589 |

Site B, val daytime:

| model | MAE kW | RMSE kW | nMAE % |
|---|---|---|---|
| B + CS feats (sin) | 2.456 | 4.131 | 4.913 |
| Model A | 2.547 | 4.109 | 5.093 |
| B k-index (sin) | 2.559 | 3.994 | 5.117 |
| B k-index (env) | 2.571 | 4.155 | 5.143 |
| B + CS feats k-index (env) | 2.572 | 4.168 | 5.144 |
| Model B | 2.587 | 4.154 | 5.174 |
| Persistence | 3.559 | 5.750 | 7.117 |
| B + CS feats (env) | 4.040 | 6.756 | 8.080 |
| A + CS feats (env) | 5.060 | 7.857 | 10.121 |

Note: Model A / B here (Colab, fresh scikit-learn) are 2.859 / 3.077 on Site A, slightly different from the 2.940 / 3.035 quoted
from the earlier run; compare variants only with the baselines in the same table.

Site A val, by sky (best variant "A + CS feats (env)" vs Model B):

| group | model | MAE kW | RMSE kW | nMAE % | n |
|---|---|---|---|---|---|
| clear | Model B | 3.917 | 5.387 | 7.833 | 71 |
| clear | A + CS feats (env) | 3.175 | 5.245 | 6.351 | 71 |
| partly_cloudy | Model B | 4.968 | 6.051 | 9.937 | 58 |
| partly_cloudy | A + CS feats (env) | 3.795 | 5.478 | 7.590 | 58 |
| overcast | Model B | 2.618 | 4.752 | 5.235 | 369 |
| overcast | A + CS feats (env) | 2.652 | 4.621 | 5.304 | 369 |

## Findings

- **k-index (predicting `kW / csp`) helps only a little**: about -0.03 to -0.04 kW vs Model B on Site A, -0.02 to -0.03 on Site B. This is within the noise of 498 rows.
- **CS features with the sin proxy** are the most consistent: Site A 3.077 -> 2.979, Site B 2.587 -> 2.456 (about -0.1 kW, -4 to -5 %). The `ci_now` feature (current clear-sky index) is the main reason.
- **The envelope (env) is fragile.** Site B's train split has no Jan-Feb rows at all (checked on Colab: 0 daytime rows in Jan-Feb 2025 for Site B), so the envelope has no values for winter and the env features are NaN/wrong exactly in the validation months, hence 4.04 and 5.06 kW. On Site A the envelope looked good (A + CS feats 2.860, but this is no better than plain Model A at 2.859).
- **Best on Site A is not a Model B variant**: Model A already scores 2.859, and clear-sky features do not improve it. The by-sky gain of "A + CS feats (env)" over Model B mostly reflects Model A vs Model B (weather features hurting in clear / partly cloudy sky), not clear-sky normalisation.
- None of this changes the ranking Model A ~ Model B >> Persistence in a way larger than the noise; the gain is real-looking only for Site B with sin features.

## Recommendation for `src/`

1. Do **not** switch the target to k = kW / csp. Gain is tiny and adds complexity (clipping, fallback).
2. Optionally integrate the cheap, robust part into `features.py` (add to `FEATURE_GROUPS["time_solar"]` or a new group, and update `CONTRACT.md` first):
   - `csp_target = Q * sin(radians(clip(target_solar_elevation, 0)))`, `csp_issue` likewise with `issue_solar_elevation`;
   - `ci_now = power_lag1 / csp_issue` (NaN if `csp_issue < 2 kW`, clipped to [0, 1.5]).
   `Q` must be fitted on the train split only and stored with the model (e.g. as a model attribute), never recomputed on val/test.
3. Do **not** use the seasonal envelope: it needs a full year of train coverage per site, which Site B does not have.
4. Because the improvement (about 0.1 kW on 498 rows) is not clearly beyond noise, confirm it with a rolling/blocked validation across more months on train before adopting it.
