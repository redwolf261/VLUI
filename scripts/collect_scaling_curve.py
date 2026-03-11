"""
collect_scaling_curve.py
------------------------
Reads output/scale/*/summary.csv files and assembles a single
output/scale/scaling_curve.csv with columns:
    n_points, index, label, isolation, mean_cri, mean_lat_ns, sprt_verdict

Usage:
    python scripts/collect_scaling_curve.py
    python scripts/collect_scaling_curve.py --scale-dir output/scale
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_parser = argparse.ArgumentParser(description="Assemble scaling curve from benchmark runs")
_parser.add_argument("--scale-dir", type=str, default="",
                     help="Directory containing n/ subdirs (default output/scale)")
_args = _parser.parse_args()

SCALE_DIR = Path(_args.scale_dir) if _args.scale_dir else PROJECT_ROOT / "output" / "scale"
OUT_CSV   = SCALE_DIR / "scaling_curve.csv"

INDEX_ORDER = ["lchi", "rtree", "buffered", "grid", "lsm", "learned"]


def _collect() -> list[dict]:
    rows: list[dict] = []

    if not SCALE_DIR.exists():
        print(f"  Scale directory not found: {SCALE_DIR}")
        return rows

    # Find all n-point subdirectories that contain a summary.csv
    subdirs = sorted(
        (d for d in SCALE_DIR.iterdir() if d.is_dir() and (d / "summary.csv").exists()),
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )

    if not subdirs:
        print(f"  No summary.csv files found under {SCALE_DIR}/*/")
        return rows

    for subdir in subdirs:
        n = int(subdir.name) if subdir.name.isdigit() else subdir.name
        summary_path = subdir / "summary.csv"
        try:
            with summary_path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows.append({
                        "n_points":    n,
                        "index":       row.get("index", ""),
                        "label":       row.get("label", ""),
                        "isolation":   row.get("isolation", ""),
                        "mean_cri":    row.get("mean_cri", ""),
                        "mean_lat_ns": row.get("mean_lat_ns", ""),
                        "sprt_verdict": row.get("sprt_verdict", ""),
                    })
        except (OSError, csv.Error) as e:
            print(f"  Warning: could not read {summary_path}: {e}")

    return rows


rows = _collect()

if not rows:
    print("\n  No data collected. Run scaling benchmarks first:")
    for n in [1000, 5000, 10000, 20000, 100000]:
        print(f"    python scripts/run_real_benchmark.py "
              f"--n-points {n} --n-queries {n // 5} "
              f"--out-dir output/scale/{n}")
    sys.exit(0)

# Save combined CSV
SCALE_DIR.mkdir(parents=True, exist_ok=True)
fieldnames = ["n_points", "index", "label", "isolation", "mean_cri", "mean_lat_ns", "sprt_verdict"]
with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Print summary table
print()
print("╔══════════════════════════════════════════════════════════════════╗")
print("║   VLUI — Scaling Curve Summary                                  ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

# Group by n_points
from itertools import groupby
rows.sort(key=lambda r: (int(r["n_points"]) if str(r["n_points"]).isdigit() else 0,
                          INDEX_ORDER.index(r["index"]) if r["index"] in INDEX_ORDER else 99))

n_vals = sorted({r["n_points"] for r in rows}, key=lambda x: int(x) if str(x).isdigit() else 0)

# Header
hdr = f"  {'n':>8}  {'Index':<20} {'Isolation':>10} {'Mean CRI':>12} {'Lat (ns)':>10} {'SPRT':>10}"
sep = "  " + "─" * (len(hdr) - 2)
print(hdr)
print(sep)

prev_n = None
for row in rows:
    if row["n_points"] != prev_n:
        if prev_n is not None:
            print(f"  {'─'*8}  {'─'*20}")
        prev_n = row["n_points"]
    try:
        iso = f"{float(row['isolation']):>10.4f}"
        cri = f"{float(row['mean_cri']):>12.6f}"
        lat = f"{float(row['mean_lat_ns']):>10.0f}"
    except (ValueError, TypeError):
        iso = cri = lat = f"{'—':>10}"
    print(f"  {str(row['n_points']):>8}  {row['label']:<20} {iso} {cri} {lat} {row['sprt_verdict']:>10}")

print(sep)

# LCHI isolation advantage vs R-tree across n
print(f"\n  LCHI isolation advantage (R-tree isolation − LCHI isolation):")
by_n: dict = {}
for row in rows:
    by_n.setdefault(row["n_points"], {})[row["index"]] = row
for n in n_vals:
    d = by_n.get(n, {})
    lchi = d.get("lchi", {})
    rtree = d.get("rtree", {})
    try:
        adv = float(rtree.get("isolation", "nan")) - float(lchi.get("isolation", "nan"))
        print(f"    n={str(n):>8}: {adv:+.4f}")
    except (ValueError, TypeError):
        print(f"    n={str(n):>8}: —")

print(f"\n  Curve written to {OUT_CSV}")
print()
