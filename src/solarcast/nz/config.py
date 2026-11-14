"""All New Zealand constants in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # repository root
RAW_DIR = ROOT / "data" / "raw" / "nz"              # cached Open-Meteo downloads

# Region centres (lat, lon). Order matters: it is the order of the map and of region_id.
REGIONS = {
    "Whangarei": (-35.725, 174.323),
    "Auckland": (-36.848, 174.763),
    "Hamilton": (-37.787, 175.279),
    "Tauranga": (-37.687, 176.165),
    "Gisborne": (-38.662, 178.018),
    "Napier": (-39.493, 176.912),
    "New Plymouth": (-39.057, 174.075),
    "Palmerston North": (-40.356, 175.611),
    "Wellington": (-41.287, 174.776),
    "Nelson": (-41.271, 173.284),
    "Christchurch": (-43.532, 172.636),
    "Queenstown": (-45.031, 168.663),
    "Dunedin": (-45.874, 170.504),
    "Invercargill": (-46.413, 168.354),
}
LOCAL_TZ = "Pacific/Auckland"

# Open-Meteo hourly variables. Radiation (shortwave, DNI, diffuse) at timestamp T is the MEAN over [T-1h, T);
# every other variable is the instantaneous value at T.
HOURLY_VARS = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "cloud_cover",
               "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "is_day"]
COLUMNS = ["ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct", "temp_c", "humidity_pct", "wind_kmh",
           "precip_mm", "is_day"]
HISTORY_START = "2022-01-01"

# The workshop's rooftop PV system and how power is derived from irradiance (stated in the write-up).
SYSTEM_KW = 50.0
PERFORMANCE_RATIO = 0.80
TEMP_COEFF = -0.004          # per degC of cell temperature above 25 degC

# Splits on the TARGET hour (UTC), half-open [start, end).
SPLITS = {
    "train": ("2022-01-01", "2025-01-01"),
    "val": ("2025-01-01", "2025-07-01"),
    "test": ("2025-07-01", "2026-11-01"),     # covers a full NZ summer
}
DAYTIME_ELEVATION_MIN = 5.0   # degrees; "daytime" rows are used for metrics

MODEL_PATH = ROOT / "models" / "nz_model.joblib"
MODEL_META = ROOT / "models" / "nz_model.json"

# EV charging scenario (NZ assumptions, same rule as the older charging module).
WORK_START, WORK_END = 8, 18   # EV plugged in 08:00-18:00 local time
EV_NEED_KWH = 40.0
CHARGER_KW = 11.0
WORKSHOP_LOAD_KW = 6.0
GRID_PRICE_NZD = 0.30
GRID_CO2_KG_PER_KWH = 0.10
