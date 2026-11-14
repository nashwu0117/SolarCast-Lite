# Forecast error and EV decision value (test period, committed model)

Daytime rows: 73746; mean daytime output 14.27 kW of 50 kW.

| Method | MAE W/m2 | MAE kW | % of 50 kW capacity | % of mean daytime output |
|---|---|---|---|---|
| SolarCast v4 | 29.55 | 1.121 | 2.24 | 7.9 |
| Persistence | 105.44 | 4.034 | 8.07 | 28.3 |
| Clear-sky persistence | 43.82 | 1.662 | 3.32 | 11.7 |
| Open-Meteo day-ahead forecast only | 49.25 | 1.878 | 3.76 | 13.2 |

Improvement of v4: {'persistence': 72.2, 'clear_sky': 32.6, 'day_ahead_forecast': 40.3}. Regions beating clear-sky persistence: 14/14. MAE by sky: {'cloud_le_60': 15.8, 'cloud_gt_60': 41.1} W/m2.

## EV charging (6818 region-days, grid energy caused by the EV, kWh/day)

| Strategy | Grid kWh/day |
|---|---|
| arrival | 18.01 |
| fixed_10_14 | 8.46 |
| persistence | 7.77 |
| clear_sky | 7.6 |
| model | 6.96 |
| perfect | 5.31 |

v4 vs fixed 10:00-14:00 window: {'kwh_per_day': 1.5, 'nzd_per_day': 0.45, 'kg_co2_per_day': 0.15}; gap to a perfect forecast: 1.65 kWh/day.
