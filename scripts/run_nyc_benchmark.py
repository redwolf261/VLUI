"""
run_nyc_benchmark.py
--------------------
Standalone script: loads the NYC TLC yellow taxi dataset, runs the full 6-index
CRI benchmark, and stores results to output/nyc_benchmark/.

Dataset: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Recommended file: yellow_tripdata_2015-01.csv (~2.3 GB, ~12M rows)
Expected columns: pickup_longitude, pickup_latitude

Usage:
    python scripts/run_nyc_benchmark.py
    python scripts/run_nyc_benchmark.py --n-points 10000 --n-queries 2000
    python scripts/run_nyc_benchmark.py --dataset path/to/yellow_tripdata_2015-01.csv
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

# ── Config (CLI args override these defaults) ─────────────────────────────────

_parser = argparse.ArgumentParser(description="VLUI NYC Taxi CRI benchmark")
_parser.add_argument("--n-points",   type=int, default=1000,  help="Insert points (default 1000)")
_parser.add_argument("--n-queries",  type=int, default=200,   help="Range queries (default 200)")
_parser.add_argument("--grid-dims",  type=str, default="2,2", help="Grid dimensions, e.g. 4,4 (default 2,2)")
_parser.add_argument("--out-dir",    type=str, default="",    help="Output directory (default output/nyc_benchmark)")
_parser.add_argument("--dataset",    type=str, default="",    help="Path to yellow_tripdata CSV")
_args = _parser.parse_args()

_gd = tuple(int(x) for x in _args.grid_dims.split(","))

DATASET_PATH = (
    Path(_args.dataset) if _args.dataset
    else PROJECT_ROOT / "data" / "nyc_taxi" / "yellow_tripdata_2015-01.csv"
)
N_POINTS    = _args.n_points
N_QUERIES   = _args.n_queries
GRID_DIMS   = _gd  # type: ignore[assignment]
WINDOW_SIZE = 50
OUT_DIR     = Path(_args.out_dir) if _args.out_dir else PROJECT_ROOT / "output" / "nyc_benchmark"

INDEX_LABELS = {
    "rtree":    "R-tree",
    "lchi":     "LCHI (VLUI ✓)",
    "buffered": "BufferedRTree",
    "grid":     "GridBTree",
    "lsm":      "LSMSpatial",
    "learned":  "LearnedGrid",
}


def _bar(pct: float, width: int = 40) -> str:
    filled = int(pct / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:5.1f}%"


# ── Load dataset ──────────────────────────────────────────────────────────────

print()
print("╔══════════════════════════════════════════════════════╗")
print("║   VLUI — Real Dataset Benchmark (NYC Taxi)           ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"\n  Dataset : {DATASET_PATH}")
print(f"  Points  : {N_POINTS:,}  |  Queries: {N_QUERIES:,}")
print("\n  Loading & normalising coordinates …", flush=True)
t_load = time.time()

loader = RealDatasetLoader.from_nyc_taxi_csv(
    DATASET_PATH,
    n_points=N_POINTS,
    n_queries=N_QUERIES,
    grid_dims=GRID_DIMS,
)
trace = loader.generate()
load_time = time.time() - t_load
total_ops = len(trace.operations)

print(f"  Source  : {loader.source_name}")
print(f"  Total ops: {total_ops:,}  ({load_time:.2f}s to load)")
print(f"  Region skew: {loader.region_point_counts}")

# ── Build indexes & measurers ─────────────────────────────────────────────────

n_regions = GRID_DIMS[0] * GRID_DIMS[1]
indexes = {
    "rtree":    NaiveRTreeIndex(dim=2, leaf_capacity=50),
    "lchi":     LCHIIndex(dim=2, grid_dims=GRID_DIMS),
    "buffered": BufferedRTreeIndex(dim=2, leaf_capacity=50, buffer_capacity=1000),
    "grid":     GridGlobalBTreeIndex(dim=2, grid_dims=GRID_DIMS),
    "lsm":      LSMSpatialIndex(dim=2, memtable_capacity=1000),
    "learned":  LearnedGridIndex(dim=2, grid_dims=GRID_DIMS),
}
measurers  = {k: CRIMeasurer(n_regions=n_regions, window_size=WINDOW_SIZE) for k in indexes}
sprts      = {k: IndexSPRT(n_regions=n_regions, delta=0.05, sigma=0.10, alpha=0.05, beta=0.05) for k in indexes}
latencies: dict[str, list[float]] = {k: [] for k in indexes}

snapshots: list[dict] = []
UPDATE_EVERY = max(30, total_ops // 200)
print(f"\n  Running benchmark …\n")
t0 = time.time()

# ── Main loop ─────────────────────────────────────────────────────────────────

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

    if (i + 1) % UPDATE_EVERY == 0 or (i + 1) == total_ops:
        elapsed = time.time() - t0
        pct = (i + 1) / total_ops * 100
        print(f"  {_bar(pct)}  op {i+1:>5}/{total_ops}", end="\r", flush=True)

        snap: dict = {"op": i + 1, "elapsed_s": round(elapsed, 3)}
        for k in indexes:
            cri = measurers[k].compute_cri()
            lats = latencies[k]
            mean_lat = sum(lats) / len(lats) if lats else 0.0
            sprts[k].update_from_matrix(cri.cri_matrix.tolist())
            snap[f"{k}_isolation"] = round(cri.isolation_score, 6)
            snap[f"{k}_mean_cri"]  = round(cri.mean_cross_cri,  8)
            snap[f"{k}_lat_ns"]    = round(mean_lat, 1)
            snap[f"{k}_sprt"]      = sprts[k].verdict
        snapshots.append(snap)

elapsed_total = time.time() - t0
print(f"\n  Done in {elapsed_total:.1f}s\n")

# ── Final results ─────────────────────────────────────────────────────────────

final: dict[str, dict] = {}
for k in indexes:
    cri = measurers[k].compute_cri()
    lats = latencies[k]
    mean_lat = sum(lats) / len(lats) if lats else 0.0
    final[k] = {
        "label":           INDEX_LABELS[k],
        "isolation":       round(cri.isolation_score, 6),
        "mean_cri":        round(cri.mean_cross_cri, 8),
        "mean_lat_ns":     round(mean_lat, 1),
        "cri_matrix":      cri.cri_matrix.tolist(),
        "sprt_verdict":    str(sprts[k].verdict).replace("SPRTDecision.", ""),
        "sprt_n_accepted": sprts[k].n_accepted,
        "sprt_n_rejected": sprts[k].n_rejected,
    }

vlui = final["lchi"]
hdr  = f"  {'Index':<20} {'Isolation':>10} {'Mean CRI':>12} {'Lat (ns)':>12} {'SPRT':>14}"
sep  = "  " + "─" * (len(hdr) - 2)
print(hdr)
print(sep)
for k, v in final.items():
    iso_delta = f"({v['isolation'] - vlui['isolation']:+.4f})" if k != "lchi" else "  ← VLUI ref"
    print(f"  {v['label']:<20} {v['isolation']:>10.4f} {iso_delta:<14}"
          f"{v['mean_cri']:>12.6f} {v['mean_lat_ns']:>12.0f}"
          f"  {v['sprt_verdict']}")
print(sep)

# ── Save results ──────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

result_json = {
    "meta": {
        "dataset":    str(DATASET_PATH),
        "source":     loader.source_name,
        "n_points":   N_POINTS,
        "n_queries":  N_QUERIES,
        "total_ops":  total_ops,
        "elapsed_s":  round(elapsed_total, 2),
        "region_counts": loader.region_point_counts,
    },
    "final": final,
    "snapshots": snapshots,
}
json_path = OUT_DIR / "results.json"
json_path.write_text(json.dumps(result_json, indent=2))

csv_path = OUT_DIR / "timeline.csv"
if snapshots:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=snapshots[0].keys())
        writer.writeheader()
        writer.writerows(snapshots)

summary_path = OUT_DIR / "summary.csv"
rows = []
for k, v in final.items():
    rows.append({
        "index":        k,
        "label":        v["label"],
        "isolation":    v["isolation"],
        "mean_cri":     v["mean_cri"],
        "mean_lat_ns":  v["mean_lat_ns"],
        "sprt_verdict": v["sprt_verdict"],
        "sprt_accepted_pairs": v["sprt_n_accepted"],
        "sprt_rejected_pairs": v["sprt_n_rejected"],
        "delta_iso_vs_vlui": round(v["isolation"] - vlui["isolation"], 6),
        "delta_cri_vs_vlui": round(v["mean_cri"]  - vlui["mean_cri"],  8),
        "delta_lat_vs_vlui": round(v["mean_lat_ns"] - vlui["mean_lat_ns"], 1),
    })
with summary_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\n  Results saved to {OUT_DIR}/")
print(f"    results.json   — full JSON (metadata + final + timeline)")
print(f"    timeline.csv   — per-window CRI/latency/SPRT snapshots")
print(f"    summary.csv    — one row per index, final metrics + deltas vs VLUI")
print()
