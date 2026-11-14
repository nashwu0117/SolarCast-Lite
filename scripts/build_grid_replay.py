"""Ship the nationwide grid's weather around the dashboard's default replay hour (data/nz/grid_replay.csv.gz).

With this file the notebook computes the replay grid in seconds without downloading anything, and judges never wait
for (or are rate-limited by) the weather API before the dashboard link appears. Other replay days are still
downloaded on demand. Rebuild after changing the cells or the replay data:  python scripts/build_grid_replay.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarcast.nz import app, config, data  # noqa: E402
from solarcast.nz.grid import GRID_REPLAY_PATH, load_cells  # noqa: E402


def main():
    svc = app.load_service()
    day = pd.Timestamp(svc.default_time()).tz_convert(config.LOCAL_TZ).normalize()
    start = (day - pd.Timedelta("2D")).tz_convert("UTC").strftime("%Y-%m-%d")
    end = (day + pd.Timedelta("2D")).tz_convert("UTC").strftime("%Y-%m-%d")
    _, cells = load_cells()
    names = [f"cell{i:03d}" for i in range(len(cells))]
    raw = data.fetch_points([(c.get("qlat", c["lat"]), c.get("qlon", c["lon"])) for c in cells], names, start, end)
    raw.to_csv(GRID_REPLAY_PATH, index=False, float_format="%.2f")
    print(f"{len(raw)} rows, {start}..{end} -> {GRID_REPLAY_PATH} ({GRID_REPLAY_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
