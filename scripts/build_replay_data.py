"""Build data/nz/replay.csv.gz: the small slice of Open-Meteo history that the dashboard replays.

It holds the TEST period (config.SPLITS["test"]) plus a few days before it (needed for the lag features),
for the 14 regions. Solar elevation/azimuth are NOT stored: they are recomputed on load (see nz/app.py).

Usage (run on Colab or any machine with internet):  PYTHONPATH=src python scripts/build_replay_data.py
Data: Open-Meteo, CC BY 4.0 (https://open-meteo.com).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarcast.nz import config, data          # noqa: E402

OUT = config.ROOT / "data" / "nz" / "replay.csv.gz"
LEAD_DAYS = 3                                    # history before the test period, for the lag features
STORED = ["region", "time_utc"] + config.COLUMNS


def build(start: str, end: str, out: Path = OUT) -> pd.DataFrame:
    df = data.fetch_history(start, end)
    df = df[STORED].copy()
    df["time_utc"] = df["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in config.COLUMNS:                    # 1 decimal is the precision Open-Meteo itself gives
        df[c] = df[c].round(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%g", compression={"method": "gzip", "compresslevel": 9})
    print(f"wrote {out}  rows={len(df)}  size={out.stat().st_size / 1e6:.2f} MB  "
          f"{df['time_utc'].min()} -> {df['time_utc'].max()}")
    return df


if __name__ == "__main__":
    t0, t1 = (pd.Timestamp(x) for x in config.SPLITS["test"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=(t0 - pd.Timedelta(days=LEAD_DAYS)).strftime("%Y-%m-%d"))
    ap.add_argument("--end", default=t1.strftime("%Y-%m-%d"), help="last day to fetch (default: end of the test period)")
    a = ap.parse_args()
    build(a.start, a.end)
