"""
compare_datasets.py
-------------------
Cross-dataset comparison: reads both Porto Taxi and NYC Taxi benchmark summaries
and prints a side-by-side table confirming LCHI leads on both datasets.

Usage:
    python scripts/compare_datasets.py
    python scripts/compare_datasets.py --porto output/real_benchmark/summary.csv \
                                       --nyc    output/nyc_benchmark/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_parser = argparse.ArgumentParser(description="Cross-dataset CRI comparison")
_parser.add_argument("--porto", type=str, default="",
                     help="Porto summary CSV (default output/real_benchmark/summary.csv)")
_parser.add_argument("--nyc",   type=str, default="",
                     help="NYC summary CSV (default output/nyc_benchmark/summary.csv)")
_parser.add_argument("--metric", type=str, default="isolation",
                     choices=["isolation", "mean_cri", "mean_lat_ns"],
                     help="Metric to compare (default: isolation)")
_args = _parser.parse_args()

PORTO_PATH = (
    Path(_args.porto) if _args.porto
    else PROJECT_ROOT / "output" / "real_benchmark" / "summary.csv"
)
NYC_PATH = (
    Path(_args.nyc) if _args.nyc
    else PROJECT_ROOT / "output" / "nyc_benchmark" / "summary.csv"
)
METRIC = _args.metric


def _load_summary(path: Path) -> dict[str, dict[str, str]]:
    """Load a summary CSV and return {index_key → row_dict}."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return {row["index"]: dict(row) for row in reader}


def _fmt(val: str, metric: str) -> str:
    try:
        f = float(val)
        if metric == "mean_lat_ns":
            return f"{f:>10.0f} ns"
        return f"{f:>10.4f}"
    except (ValueError, TypeError):
        return f"{val:>13}"


INDEX_ORDER = ["lchi", "rtree", "buffered", "grid", "lsm", "learned"]

porto = _load_summary(PORTO_PATH)
nyc   = _load_summary(NYC_PATH)

# ── Print comparison table ────────────────────────────────────────────────────

print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║   VLUI — Cross-Dataset Comparison                           ║")
print("╚══════════════════════════════════════════════════════════════╝")

if not porto and not nyc:
    print("\n  No results found. Run run_real_benchmark.py and run_nyc_benchmark.py first.")
    sys.exit(0)

metric_label = {"isolation": "Isolation Score", "mean_cri": "Mean Cross-CRI",
                "mean_lat_ns": "Mean Latency (ns)"}[METRIC]

print(f"\n  Metric: {metric_label}\n")
print(f"  {'Index':<20} {'Porto Taxi':>13}  {'NYC Taxi':>13}  {'LCHI leads?':>12}")
print(f"  {'─'*62}")

lchi_porto = float(porto.get("lchi", {}).get(METRIC, "nan") or "nan")
lchi_nyc   = float(nyc.get("lchi", {}).get(METRIC, "nan") or "nan")

for idx_key in INDEX_ORDER:
    label = (porto.get(idx_key) or nyc.get(idx_key) or {}).get("label", idx_key)

    p_val = porto.get(idx_key, {}).get(METRIC, "—")
    n_val = nyc.get(idx_key, {}).get(METRIC, "—")

    p_fmt = _fmt(p_val, METRIC) if p_val != "—" else f"{'—':>13}"
    n_fmt = _fmt(n_val, METRIC) if n_val != "—" else f"{'—':>13}"

    if idx_key == "lchi":
        leads = "← VLUI ref"
    else:
        try:
            p_better = float(p_val) > lchi_porto if porto else None
            n_better = float(n_val) > lchi_nyc   if nyc   else None
            both = [x for x in [p_better, n_better] if x is not None]
            leads = "YES ✓" if all(both) else ("partial" if any(both) else "NO")
        except (ValueError, TypeError):
            leads = "—"

    print(f"  {label:<20} {p_fmt}  {n_fmt}  {leads:>12}")

print(f"  {'─'*62}")

if porto and nyc:
    lchi_iso_porto  = float(porto.get("lchi", {}).get("isolation", "nan") or "nan")
    rtree_iso_porto = float(porto.get("rtree", {}).get("isolation", "nan") or "nan")
    lchi_iso_nyc    = float(nyc.get("lchi", {}).get("isolation", "nan") or "nan")
    rtree_iso_nyc   = float(nyc.get("rtree", {}).get("isolation", "nan") or "nan")
    print(f"\n  LCHI isolation advantage (R-tree - LCHI):")
    print(f"    Porto: {rtree_iso_porto - lchi_iso_porto:+.4f}")
    print(f"    NYC  : {rtree_iso_nyc   - lchi_iso_nyc:+.4f}")

print()

missing = []
if not porto:
    missing.append(f"  Porto : {PORTO_PATH}")
if not nyc:
    missing.append(f"  NYC   : {NYC_PATH}")
if missing:
    print("  Missing result files (run the benchmark scripts first):")
    for m in missing:
        print(m)
    print()
