"""
Experiment runner: loads config, runs all (index × workload) combinations,
and produces structured results.

Usage:
    python -m src.benchmark.run_experiment configs/experiment_baseline.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
import numpy as np
import pandas as pd

from src.generators.base import BaseGenerator
from src.generators.uniform import UniformRandomGenerator
from src.generators.zipf_spatial import ZipfSpatialGenerator
from src.generators.moving_hotspot import MovingHotspotGenerator, HotspotPath
from src.generators.adversarial import AdversarialBurstGenerator
from src.indexes.base import SpatialIndex
from src.indexes.rtree_index import NaiveRTreeIndex
from src.indexes.lchi_index import LCHIIndex
from src.indexes.buffered_rtree import BufferedRTreeIndex
from src.indexes.grid_btree import GridGlobalBTreeIndex
from src.indexes.lsm_spatial import LSMSpatialIndex
from src.benchmark.executor import BenchmarkExecutor, BenchmarkResult


GENERATOR_MAP: dict[str, type[BaseGenerator]] = {
    "uniform": UniformRandomGenerator,
    "zipf": ZipfSpatialGenerator,
    "moving_hotspot": MovingHotspotGenerator,
    "adversarial": AdversarialBurstGenerator,
}


def build_generator(workload_cfg: dict[str, Any], space_cfg: dict[str, Any], seed: int) -> BaseGenerator:
    """Construct a generator from config."""
    gen_type: str = workload_cfg["generator"]
    params: dict[str, Any] = dict(workload_cfg.get("params", {}))

    # Inject space config
    params["dim"] = space_cfg["dim"]
    params["space_lo"] = space_cfg["lo"]
    params["space_hi"] = space_cfg["hi"]
    params["grid_dims"] = tuple(space_cfg["grid_dims"])  # type: ignore[arg-type]
    params["seed"] = seed

    # Handle enum conversions
    if gen_type == "moving_hotspot" and "path_type" in params:
        params["path_type"] = HotspotPath(params["path_type"])

    cls = GENERATOR_MAP[gen_type]
    return cls(**params)  # type: ignore[arg-type]


def build_index(index_cfg: dict[str, Any], space_cfg: dict[str, Any]) -> SpatialIndex:
    """Construct an index from config."""
    idx_type: str = index_cfg["type"]
    params: dict[str, Any] = dict(index_cfg.get("params", {}))

    if idx_type == "rtree":
        return NaiveRTreeIndex(dim=int(space_cfg["dim"]), **params)  # type: ignore[arg-type]
    elif idx_type == "lchi":
        grid_dims = tuple(params.get("grid_dims", space_cfg["grid_dims"]))  # type: ignore[arg-type]
        return LCHIIndex(
            dim=int(space_cfg["dim"]),
            space_lo=float(space_cfg["lo"]),
            space_hi=float(space_cfg["hi"]),
            grid_dims=grid_dims,
        )
    elif idx_type == "buffered_rtree":
        return BufferedRTreeIndex(
            dim=int(space_cfg["dim"]),
            leaf_capacity=int(params.get("leaf_capacity", 50)),
            buffer_capacity=int(params.get("buffer_capacity", 1000)),
        )
    elif idx_type == "grid_btree":
        grid_dims = tuple(params.get("grid_dims", space_cfg["grid_dims"]))  # type: ignore[arg-type]
        return GridGlobalBTreeIndex(
            dim=int(space_cfg["dim"]),
            space_lo=float(space_cfg["lo"]),
            space_hi=float(space_cfg["hi"]),
            grid_dims=grid_dims,
        )
    elif idx_type == "lsm_spatial":
        return LSMSpatialIndex(
            dim=int(space_cfg["dim"]),
            memtable_capacity=int(params.get("memtable_capacity", 1000)),
            level_ratio=int(params.get("level_ratio", 4)),
            max_levels=int(params.get("max_levels", 5)),
        )
    else:
        raise ValueError(f"Unknown index type: {idx_type}")


def run_experiment(config_path: str) -> list[BenchmarkResult]:
    """Run full experiment from config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    space_cfg = cfg["space"]
    cri_cfg = cfg["cri"]
    seed = exp_cfg["seed"]
    output_dir = Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    executor = BenchmarkExecutor(
        cri_window_size=cri_cfg["window_size"],
        show_progress=True,
    )

    all_results: list[BenchmarkResult] = []
    summaries: list[dict[str, Any]] = []

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_cfg['name']}")
    print(f"  {exp_cfg['description']}")
    print(f"{'='*70}\n")

    for wl_cfg in cfg["workloads"]:
        wl_name = wl_cfg["name"]
        print(f"\n-- Workload: {wl_name} --")

        gen = build_generator(wl_cfg, space_cfg, seed)
        trace = gen.generate()

        print(f"   Generated {trace.n_ops} operations across {trace.n_regions} regions")
        print(f"   Update distribution: {trace.update_rate_per_region()}")

        for idx_cfg in cfg["indexes"]:
            index = build_index(idx_cfg, space_cfg)
            print(f"\n   Index: {index.name}")

            result = executor.run(index, trace, workload_name=wl_name)

            all_results.append(result)
            summary = result.summary()
            summaries.append(summary)

            print(f"   |  Mean query latency:  {summary['mean_query_ns']:.0f} ns")
            print(f"   |  P99 query latency:   {summary['p99_query_ns']:.0f} ns")
            print(f"   |  Isolation score:     {summary['isolation_score']:.6f}")
            print(f"   |  Mean cross-CRI:      {summary['mean_cross_cri']:.6f}")
            print(f"   +- eps-isolated (0.01): {summary['epsilon_isolated_0.01']}")

    # Save results
    results_df = pd.DataFrame(summaries)
    results_path = output_dir / f"{exp_cfg['name']}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n\nResults saved to {results_path}")

    # Save CRI matrices
    for result in all_results:
        safe_name = f"{result.workload_name}_{result.index_name}".replace(" ", "_")
        cri_path = output_dir / f"{safe_name}_cri_matrix.npy"
        np.save(cri_path, result.cri_result.cri_matrix)

    print(f"\n{'='*70}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")

    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.benchmark.run_experiment <config.yaml>")
        sys.exit(1)
    run_experiment(sys.argv[1])
