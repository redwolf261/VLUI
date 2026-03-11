"""
run_grid_sweep.py
-----------------
Grid dimension sweep: runs all 6 VLUI indexes across grid_dims ∈ [(2,2), (4,4), (8,8)]
using the Porto Taxi dataset with n_points=10K per grid.

The key result: LCHI isolation advantage over R-tree grows with m (number of partitions),
because LCHI's per-partition walls become tighter while global-structure indexes degrade.

Usage:
    python scripts/run_grid_sweep.py
    python scripts/run_grid_sweep.py --n-points 5000
    python scripts/run_grid_sweep.py --n-points 10000 --out-dir output/grid_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.real_dataset import RealDatasetLoader
from src.indexes.rtree_index import NaiveRTreeIndex
from src.indexes.lchi_index import LCHIIndex
from src.indexes.buffered_rtree import BufferedRTreeIndex
from src.indexes.grid_btree import GridGlobalBTreeIndex
from src.indexes.lsm_spatial import LSMSpatialIndex
from src.indexes.learned_grid import LearnedGridIndex
from src.metrics.cri import CRIMeasurer
from src.models.spatial import OpType
from src.analysis.sprt import IndexSPRT

# ── CLI args ──────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="VLUI grid-dimension sweep")
_parser.add_argument("--n-points",  type=int, default=10000,  help="Insert points per sweep leg (default 10000)")
_parser.add_argument("--n-queries", type=int, default=2000,   help="Range queries per sweep leg (default 2000)")
_parser.add_argument("--out-dir",   type=str, default="",     help="Output directory (default output/grid_sweep)")
_parser.add_argument("--dataset",   type=str, default="",     help="Path to Porto train.csv (default data/porto_taxi/train.csv)")
_args = _parser.parse_args()

DATASET_PATH = Path(_args.dataset) if _args.dataset else PROJECT_ROOT / "data" / "porto_taxi" / "train.csv"
N_POINTS     = _args.n_points
N_QUERIES    = _args.n_queries
WINDOW_SIZE  = 100
OUT_DIR      = Path(_args.out_dir) if _args.out_dir else PROJECT_ROOT / "output" / "grid_sweep"

GRID_CONFIGS = [(2, 2), (4, 4), (8, 8)]

INDEX_LABELS = {
    "rtree":    "R-tree",
    "lchi":     "LCHI (VLUI ✓)",
    "buffered": "BufferedRTree",
    "grid":     "GridBTree",
    "lsm":      "LSMSpatial",
    "learned":  "LearnedGrid",
}


def _bar(pct: float, width: int = 35) -> str:
    filled = int(pct / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:5.1f}%"


def _run_one_grid(grid_dims: tuple[int, int]) -> dict:
    """Run the full 6-index benchmark for a single grid configuration."""
    label = f"{grid_dims[0]}x{grid_dims[1]}"
    n_regions = grid_dims[0] * grid_dims[1]

    print(f"\n  ┌─ Grid {label}  ({n_regions} regions) ─────────────────────────────")

    t_load = time.time()
    loader = RealDatasetLoader.from_porto_csv(
        DATASET_PATH,
        n_points=N_POINTS,
        n_queries=N_QUERIES,
        grid_dims=grid_dims,
    )
    trace = loader.generate()
    load_time = time.time() - t_load
    total_ops = len(trace.operations)
    print(f"  │  Loaded {total_ops:,} ops  ({load_time:.2f}s)  skew: {loader.region_point_counts}")

    indexes = {
        "rtree":    NaiveRTreeIndex(dim=2, leaf_capacity=50),
        "lchi":     LCHIIndex(dim=2, grid_dims=grid_dims),
        "buffered": BufferedRTreeIndex(dim=2, leaf_capacity=50, buffer_capacity=1000),
        "grid":     GridGlobalBTreeIndex(dim=2, grid_dims=grid_dims),
        "lsm":      LSMSpatialIndex(dim=2, memtable_capacity=1000),
        "learned":  LearnedGridIndex(dim=2, grid_dims=grid_dims),
    }
    measurers = {k: CRIMeasurer(n_regions=n_regions, window_size=WINDOW_SIZE) for k in indexes}
    sprts     = {k: IndexSPRT(n_regions=n_regions, delta=0.05, sigma=0.10, alpha=0.05, beta=0.05) for k in indexes}
    latencies: dict[str, list[float]] = {k: [] for k in indexes}

    update_every = max(50, total_ops // 100)
    t0 = time.time()

    for i, op in enumerate(trace.operations):
        if op.op_type == OpType.INSERT and op.point is not None:
            for k, idx in indexes.items():
                s = idx.insert(op.point)
                measurers[k].record_update(op.region_id, op.timestamp, s.structural_mutations)

        elif op.op_type == OpType.RANGE_QUERY and op.query_region is not None:
            for k, idx in indexes.items():
                _, s = idx.range_query(op.query_region)
                latencies[k].append(s.latency_ns)
                measurers[k].record_query(op.region_id, op.timestamp, s.latency_ns)

        if (i + 1) % update_every == 0 or (i + 1) == total_ops:
            pct = (i + 1) / total_ops * 100
            print(f"  │  {_bar(pct)}  {i+1}/{total_ops}", end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"  │  Done in {elapsed:.1f}s" + " " * 30)

    results: dict[str, dict] = {}
    for k in indexes:
        cri = measurers[k].compute_cri()
        sprts[k].update_from_matrix(cri.cri_matrix.tolist())
        lats = latencies[k]
        mean_lat = sum(lats) / len(lats) if lats else 0.0
        results[k] = {
            "label":           INDEX_LABELS[k],
            "isolation":       round(cri.isolation_score, 6),
            "mean_cri":        round(cri.mean_cross_cri, 8),
            "mean_lat_ns":     round(mean_lat, 1),
            "n_windows":       cri.n_windows,
            "sprt_verdict":    str(sprts[k].verdict).replace("SPRTDecision.", ""),
        }

    # Print mini table
    vlui_iso = results["lchi"]["isolation"]
    print(f"  │  {'Index':<20} {'Isolation':>10}  {'Δ vs LCHI':>12}  {'Lat (ns)':>10}")
    print(f"  │  {'─'*57}")
    for k, v in results.items():
        delta = f"({v['isolation'] - vlui_iso:+.4f})" if k != "lchi" else "← VLUI ref"
        print(f"  │  {v['label']:<20} {v['isolation']:>10.4f}  {delta:<12}  {v['mean_lat_ns']:>10.0f}")
    print(f"  └{'─'*57}")

    return {
        "grid_label": label,
        "n_regions": n_regions,
        "n_points": N_POINTS,
        "n_queries": N_QUERIES,
        "elapsed_s": round(elapsed, 2),
        "indexes": results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

print()
print("╔══════════════════════════════════════════════════════╗")
print("║   VLUI — Grid Dimension Sweep  (2×2 → 4×4 → 8×8)    ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"\n  Dataset : {DATASET_PATH}")
print(f"  Points  : {N_POINTS:,}  |  Queries: {N_QUERIES:,} per grid config")

sweep_results: dict[str, dict] = {}
for gd in GRID_CONFIGS:
    key = f"{gd[0]}x{gd[1]}"
    sweep_results[key] = _run_one_grid(gd)

# ── Verify hypothesis: LCHI isolation advantage grows with m ──────────────────

print("\n  LCHI isolation advantage vs R-tree across grids:")
for key, res in sweep_results.items():
    lchi_iso  = res["indexes"]["lchi"]["isolation"]
    rtree_iso = res["indexes"]["rtree"]["isolation"]
    advantage = rtree_iso - lchi_iso
    print(f"    {key:>4}:  R-tree={rtree_iso:.4f},  LCHI={lchi_iso:.4f},  "
          f"advantage={advantage:+.4f}")

# ── Save output ───────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

# JSON — full results keyed by "2x2", "4x4", "8x8"
json_path = OUT_DIR / "sweep_results.json"
json_path.write_text(json.dumps(sweep_results, indent=2))

# CSV — one row per (grid, index) combination
csv_path = OUT_DIR / "sweep_summary.csv"
rows = []
for key, res in sweep_results.items():
    lchi_iso = res["indexes"]["lchi"]["isolation"]
    for idx_key, v in res["indexes"].items():
        rows.append({
            "grid":             key,
            "n_regions":        res["n_regions"],
            "index":            idx_key,
            "label":            v["label"],
            "isolation":        v["isolation"],
            "mean_cri":         v["mean_cri"],
            "mean_lat_ns":      v["mean_lat_ns"],
            "sprt_verdict":     v["sprt_verdict"],
            "delta_iso_vs_lchi": round(v["isolation"] - lchi_iso, 6),
        })

with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\n  Results saved to {OUT_DIR}/")
print(f"    sweep_results.json  — full results keyed by grid config")
print(f"    sweep_summary.csv   — one row per (grid, index)")
print()
