"""
run_window_sweep.py
-------------------
Empirical validation of Lemma 3 (OLS-CRI Consistency):
Shows CRI estimate stabilises as window size W grows, consistent with
the O(W^{-1}) bias bound proved in the paper.

Strategy: run the LCHI benchmark ONCE at 100K ops, recording all ops into
a master CRIMeasurer with the finest window (w=2). Then sweep W ∈ [10, 20,
50, 100, 200, 500, 1000] by cloning only the recorded event lists into fresh
CRIMeasurer instances with each target window size and calling compute_cri().
This avoids re-running the benchmark 7 times.

Output:
    output/window_sweep/window_results.csv
        columns: window_size, mean_cross_cri, isolation_score, n_windows

Usage:
    python scripts/run_window_sweep.py
    python scripts/run_window_sweep.py --n-points 50000  # quicker sanity check
    python scripts/run_window_sweep.py --out-dir output/window_sweep --alpha 1.5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.zipf_spatial import ZipfSpatialGenerator
from src.indexes.lchi_index import LCHIIndex
from src.metrics.cri import CRIMeasurer
from src.models.spatial import OpType

# ── CLI args ──────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="VLUI window-size sweep (Lemma 3 validation)")
_parser.add_argument("--n-points",  type=int,   default=100_000, help="Insert points (default 100000)")
_parser.add_argument("--n-queries", type=int,   default=20_000,  help="Range queries (default 20000)")
_parser.add_argument("--alpha",     type=float, default=1.5,     help="Zipf skew (default 1.5)")
_parser.add_argument("--grid-dims", type=str,   default="2,2",   help="Grid dimensions (default 2,2)")
_parser.add_argument("--out-dir",   type=str,   default="",      help="Output directory")
_parser.add_argument("--seed",      type=int,   default=42,      help="Random seed (default 42)")
_args = _parser.parse_args()

_gd = tuple(int(x) for x in _args.grid_dims.split(","))

N_POINTS    = _args.n_points
N_QUERIES   = _args.n_queries
ALPHA       = _args.alpha
GRID_DIMS   = _gd  # type: ignore[assignment]
SEED        = _args.seed
OUT_DIR     = Path(_args.out_dir) if _args.out_dir else PROJECT_ROOT / "output" / "window_sweep"

# W values to sweep — must have N_POINTS large enough to yield ≥10 windows at each W.
# At 100K total ops and W=1000: 100000/1000 = 100 windows (stable).
WINDOW_SIZES = [10, 20, 50, 100, 200, 500, 1000]

N_REGIONS = GRID_DIMS[0] * GRID_DIMS[1]


def _bar(pct: float, width: int = 40) -> str:
    filled = int(pct / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:5.1f}%"


# ── Generate workload ─────────────────────────────────────────────────────────

print()
print("╔══════════════════════════════════════════════════════╗")
print("║   VLUI — Window Sweep  (Lemma 3 Empirical Valid.)    ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"\n  n_points={N_POINTS:,}  n_queries={N_QUERIES:,}  alpha={ALPHA}"
      f"  grid={GRID_DIMS[0]}x{GRID_DIMS[1]}")
print("\n  Generating Zipf workload …", flush=True)

gen = ZipfSpatialGenerator(
    n_points=N_POINTS,
    n_queries=N_QUERIES,
    alpha=ALPHA,
    grid_dims=GRID_DIMS,
    seed=SEED,
)
trace = gen.generate()
total_ops = len(trace.operations)
print(f"  Generated {total_ops:,} ops")

# ── Single benchmark run with master measurer (finest window = 2) ────────────

print("\n  Running LCHI benchmark …")
index   = LCHIIndex(dim=2, grid_dims=GRID_DIMS)
master  = CRIMeasurer(n_regions=N_REGIONS, window_size=2)
latencies: list[float] = []

update_every = max(50, total_ops // 100)
t0 = time.time()

for i, op in enumerate(trace.operations):
    if op.op_type == OpType.INSERT and op.point is not None:
        s = index.insert(op.point)
        master.record_update(op.region_id, op.timestamp, s.structural_mutations)

    elif op.op_type == OpType.RANGE_QUERY and op.query_region is not None:
        _, s = index.range_query(op.query_region)
        latencies.append(s.latency_ns)
        master.record_query(op.region_id, op.timestamp, s.latency_ns)

    if (i + 1) % update_every == 0 or (i + 1) == total_ops:
        pct = (i + 1) / total_ops * 100
        print(f"  {_bar(pct)}  {i+1}/{total_ops}", end="\r", flush=True)

elapsed = time.time() - t0
print(f"\n  Benchmark done in {elapsed:.1f}s"
      f"  ({len(master._updates):,} updates, {len(master._queries):,} queries recorded)")

# ── Window sweep: clone event lists, vary window_size ────────────────────────

print("\n  Sweeping window sizes …\n")
print(f"  {'W':>6}  {'n_windows':>10}  {'isolation':>12}  {'mean_cross_cri':>16}")
print(f"  {'─'*54}")

results: list[dict] = []

for W in WINDOW_SIZES:
    clone = CRIMeasurer(n_regions=N_REGIONS, window_size=W)
    # Copy recorded event lists — shallow copy is safe (records are frozen dataclasses)
    clone._updates = list(master._updates)
    clone._queries = list(master._queries)

    cri = clone.compute_cri()
    row = {
        "window_size":    W,
        "n_windows":      cri.n_windows,
        "isolation_score": round(cri.isolation_score,  6),
        "mean_cross_cri":  round(cri.mean_cross_cri,   8),
    }
    results.append(row)
    print(f"  {W:>6}  {cri.n_windows:>10}  {cri.isolation_score:>12.6f}  {cri.mean_cross_cri:>16.8f}")

print(f"  {'─'*54}")

# ── Verify: isolation stabilises as W grows ───────────────────────────────────

if len(results) >= 4:
    early_var = sum(
        (r["isolation_score"] - results[0]["isolation_score"]) ** 2
        for r in results[:3]
    ) / 3
    late_var  = sum(
        (r["isolation_score"] - results[-1]["isolation_score"]) ** 2
        for r in results[-3:]
    ) / 3
    print(f"\n  Variance (early W): {early_var:.6f}")
    print(f"  Variance (late  W): {late_var:.6f}")
    if late_var < early_var:
        print("  ✓ Confirmed: isolation estimate stabilises as W grows (consistent with Lemma 3)")
    else:
        print("  ! Late variance not smaller — consider increasing n_points for cleaner signal")

# ── Save output ───────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = OUT_DIR / "window_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n  Results saved to {OUT_DIR}/window_results.csv")
print(f"    columns: window_size, n_windows, isolation_score, mean_cross_cri")
print()
