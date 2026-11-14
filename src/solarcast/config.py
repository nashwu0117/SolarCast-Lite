"""Shared constants. Every module imports from here; do not hard-code these elsewhere."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw"          # downloaded CSVs are cached here (gitignored)
MODELS_DIR = REPO_ROOT / "models"              # saved models (gitignored)

# Dataset: Two-Site Solar PV Inverter & Weather Dataset (Romania), CC BY 4.0
# https://github.com/upb-nlp/PV-Generation-Dataset
DATASET_BASE_URL = "https://raw.githubusercontent.com/upb-nlp/PV-Generation-Dataset/main"
SITES = {
    "A": {"csv": "site_a/site_a_hourly_weather.csv", "capacity_kw": 50.0},
    "B": {"csv": "site_b/site_b_hourly_weather.csv", "capacity_kw": 50.0},
}
CAPACITY_KW = 50.0
LOCAL_TZ = "Europe/Bucharest"

# Columns kept by data.load_site (all others in the CSV are dropped).
TARGET_COL = "active_power_kw"
WEATHER_COLS = [
    "temp_c", "humidity_pct", "cloud_cover_pct", "wind_speed_kmh",
    "precip_mm", "ghi_wm2", "dni_wm2", "dhi_wm2",
]
SOLAR_COLS = ["solar_elevation_deg", "solar_azimuth_deg"]
KEEP_COLS = [TARGET_COL, "capacity_factor", "is_generating"] + WEATHER_COLS + SOLAR_COLS

# Main evaluation is daytime only: target-hour sun elevation above this (dataset convention).
DAYTIME_ELEVATION_MIN = 5.0

# Time-based splits on the TARGET hour (UTC), half-open [start, end). Never shuffle.
SPLITS = {
    "train": ("2025-01-01", "2026-01-01"),
    "val":   ("2026-01-01", "2026-03-01"),
    "test":  ("2026-03-01", "2026-07-01"),
}

# Sky-condition groups for error analysis, by target-hour cloud cover (%), [low, high).
SKY_BINS = {"clear": (0, 30), "partly_cloudy": (30, 70), "overcast": (70, 101)}

RANDOM_STATE = 42
