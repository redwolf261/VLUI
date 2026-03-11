"""
Generate a small but realistic sample Porto-Taxi-format CSV for testing.
Produces data/sample_porto.csv with 500 rows — enough for n_points=200.

Usage:
    python scripts/gen_sample_porto.py
"""

from __future__ import annotations
import csv
import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "data" / "sample_porto.csv"

# Porto bounding box (lon_min, lat_min, lon_max, lat_max)
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -8.73, 41.11, -8.57, 41.24

RNG = random.Random(2026)

def _rand_point() -> tuple[float, float]:
    """Random point inside Porto with slight clustering near the city centre."""
    cx, cy = -8.615, 41.158  # rough Porto centre
    # 70 % chance: Gaussian cluster around centre (σ ≈ 0.04°)
    if RNG.random() < 0.70:
        lon = cx + RNG.gauss(0, 0.04)
        lat = cy + RNG.gauss(0, 0.025)
    else:
        lon = RNG.uniform(LON_MIN, LON_MAX)
        lat = RNG.uniform(LAT_MIN, LAT_MAX)
    # clip to bounds
    lon = max(LON_MIN, min(LON_MAX, lon))
    lat = max(LAT_MIN, min(LAT_MAX, lat))
    return round(lon, 6), round(lat, 6)

def _make_trip(trip_id: int) -> dict:
    """Simulate a taxi trip as a short polyline (2–8 steps, ~15s intervals)."""
    n_steps = RNG.randint(2, 8)
    start_lon, start_lat = _rand_point()
    polyline = [[start_lon, start_lat]]
    lon, lat = start_lon, start_lat
    for _ in range(n_steps - 1):
        # small drift each step (≈ 150 m per 15 s at city speed)
        lon += RNG.gauss(0, 0.002)
        lat += RNG.gauss(0, 0.0015)
        lon = max(LON_MIN, min(LON_MAX, round(lon, 6)))
        lat = max(LAT_MIN, min(LAT_MAX, round(lat, 6)))
        polyline.append([lon, lat])

    # simulate a realistic base timestamp (2013-07-01 00:00 UTC)
    ts_base = 1372636800
    timestamp = ts_base + RNG.randint(0, 86400 * 180)

    return {
        "TRIP_ID":      str(1000000 + trip_id),
        "CALL_TYPE":    RNG.choice(["A", "B", "C"]),
        "ORIGIN_CALL":  str(RNG.randint(10000, 99999)) if RNG.random() < 0.3 else "",
        "ORIGIN_STAND": str(RNG.randint(1, 63)) if RNG.random() < 0.4 else "",
        "TAXI_ID":      str(20000000 + RNG.randint(0, 400)),
        "TIMESTAMP":    str(timestamp),
        "DAY_TYPE":     "A",
        "MISSING_DATA": "False",
        "POLYLINE":     json.dumps(polyline),
    }

FIELDNAMES = [
    "TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND",
    "TAXI_ID", "TIMESTAMP", "DAY_TYPE", "MISSING_DATA", "POLYLINE",
]

N_ROWS = 500

def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i in range(N_ROWS):
            writer.writerow(_make_trip(i))
    print(f"Wrote {N_ROWS} rows → {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size:,} bytes")

if __name__ == "__main__":
    main()
