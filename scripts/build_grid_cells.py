"""Choose the land cells of the nationwide grid once and save them to data/nz/grid_cells.json.

A 0.5-degree grid covers the New Zealand box (main islands and Stewart Island). For every cell we ask Open-Meteo's
elevation service (Copernicus DEM, sea = 0 m) for 5 sample points: the centre and four points 0.17 degrees from it.
A cell is land when at least 2 samples are above sea level (this drops open sea and tiny offshore islets). The
weather for a cell is downloaded at the mean of its land samples (qlat/qlon), so coastal cells are not sea points.

Run (Colab or any machine with internet):  python scripts/build_grid_cells.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarcast.nz.service import NZ_BOX  # noqa: E402

STEP = 0.5
OFF = 0.17
URL = "https://api.open-meteo.com/v1/elevation"
OUT = ROOT / "data" / "nz" / "grid_cells.json"


def elevations(points):
    out = []
    for i in range(0, len(points), 100):                      # the service takes up to 100 points per request
        chunk = points[i:i + 100]
        params = {"latitude": ",".join(f"{p[0]:.4f}" for p in chunk),
                  "longitude": ",".join(f"{p[1]:.4f}" for p in chunk)}
        for k in range(5):                                    # patient retry: timeouts and rate limits
            try:
                r = requests.get(URL, params=params, timeout=120)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(10 * (k + 1))
        r.raise_for_status()
        out += r.json()["elevation"]
        time.sleep(1)
    return out


def main():
    lats = np.arange(NZ_BOX[0] + STEP / 2, NZ_BOX[1], STEP)
    lons = np.arange(NZ_BOX[2] + STEP / 2, NZ_BOX[3], STEP)
    centres = [(round(float(a), 3), round(float(o), 3)) for a in lats for o in lons]
    offsets = [(0, 0), (OFF, OFF), (OFF, -OFF), (-OFF, OFF), (-OFF, -OFF)]
    samples = [(a + da, o + do) for a, o in centres for da, do in offsets]
    elev = elevations(samples)
    cells = []
    for k, (a, o) in enumerate(centres):
        land = [samples[k * 5 + j] for j in range(5) if (elev[k * 5 + j] or 0) > 0]
        if len(land) >= 2:
            cells.append({"lat": a, "lon": o, "qlat": round(float(np.mean([p[0] for p in land])), 3),
                          "qlon": round(float(np.mean([p[1] for p in land])), 3)})
    OUT.write_text(json.dumps({"step": STEP, "source": "Open-Meteo elevation API (Copernicus DEM), >= 2 of 5 samples "
                                                       "above sea level", "cells": cells}, indent=0))
    print(f"{len(cells)} land cells of {len(centres)} -> {OUT}")


if __name__ == "__main__":
    main()
