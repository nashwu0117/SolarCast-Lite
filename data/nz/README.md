# data/nz/replay.csv.gz

Hourly weather for the 14 New Zealand regions of SolarCast NZ, covering the **test period**
(2025-07-01 to 2026-11-01) plus 3 days before it (needed for the lag features). The dashboard replays it so that
nobody has to download five years of history. Rebuild it with `python scripts/build_replay_data.py`.

- Source: [Open-Meteo](https://open-meteo.com) Historical Weather API (ERA5-based reanalysis), https://archive-api.open-meteo.com/v1/archive
- License: CC BY 4.0. Attribution: "Weather data by Open-Meteo.com" (https://open-meteo.com/en/license).
- Columns: `region, time_utc` (ISO, UTC), `ghi_wm2, dni_wm2, dhi_wm2` (W/m2, mean over the PRECEDING hour),
  `cloud_cover_pct, temp_c, humidity_pct, wind_kmh, precip_mm, is_day` (instantaneous at the timestamp).
  Solar elevation/azimuth are not stored; they are recomputed on load.
- This is modelled weather, not measurements from real solar plants. PV power is derived from irradiance.
