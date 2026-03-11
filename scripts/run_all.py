"""
Reproducible experiment runner.

Runs the full experiment suite with deterministic seeds, generates
all figures, and produces a structured results directory.

Usage:
    python scripts/run_all.py                     # full suite
    python scripts/run_all.py --quick             # quick smoke test (reduced sizes)
    python scripts/run_all.py --config <path>     # custom config
    python scripts/run_all.py --skip-plots        # data only, no figures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.benchmark.run_experiment import run_experiment
from src.benchmark.visualize import generate_all_figures, build_summary_table


def quick_config(output_dir: Path) -> Path:
    """Generate a quick-test config with reduced workload sizes."""
    import yaml

    config = {
        "experiment": {
            "name": "quick_test",
            "description": "Quick smoke test with reduced sizes",
            "seed": 42,
            "output_dir": str(output_dir / "results"),
        },
        "space": {
            "dim": 2,
            "lo": 0.0,
            "hi": 1.0,
            "grid_dims": [4, 4],
        },
        "cri": {
            "window_size": 200,
            "epsilon": 0.01,
        },
        "indexes": [
            {"type": "rtree", "params": {"leaf_capacity": 50}},
            {"type": "lchi", "params": {"grid_dims": [4, 4]}},
            {"type": "buffered_rtree", "params": {"leaf_capacity": 50, "buffer_capacity": 200}},
            {"type": "grid_btree", "params": {"grid_dims": [4, 4]}},
            {"type": "lsm_spatial", "params": {"memtable_capacity": 500}},
        ],
        "workloads": [
            {
                "name": "uniform",
                "generator": "uniform",
                "params": {"n_points": 5000, "n_queries": 500, "query_radius": 0.05},
            },
            {
                "name": "zipf_alpha_2.0",
                "generator": "zipf",
                "params": {"n_points": 5000, "n_queries": 500, "alpha": 2.0, "query_radius": 0.05},
            },
            {
                "name": "adversarial_burst",
                "generator": "adversarial",
                "params": {
                    "n_points": 5000,
                    "n_queries": 500,
                    "attack_region_id": 0,
                    "burst_size": 250,
                    "n_bursts": 5,
                    "cluster_tightness": 0.001,
                    "warmup_fraction": 0.2,
                },
            },
        ],
    }

    config_path = output_dir / "quick_test_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Run VLUI experiments with full reproducibility"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to experiment config YAML (default: configs/experiment_baseline.yaml)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick smoke test with reduced workload sizes"
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip figure generation"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: output/<timestamp>)"
    )
    args = parser.parse_args()

    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "output" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Project root:     {PROJECT_ROOT}")

    # Determine config
    if args.quick:
        config_path = quick_config(output_dir)
        print(f"Using quick-test config: {config_path}")
    elif args.config:
        config_path = Path(args.config)
    else:
        config_path = PROJECT_ROOT / "configs" / "experiment_baseline.yaml"

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    # Run experiment
    print(f"\nRunning experiment from: {config_path}")
    start = time.time()
    results = run_experiment(str(config_path))
    elapsed = time.time() - start

    print(f"\nExperiment completed in {elapsed:.1f}s")
    print(f"Total runs: {len(results)}")

    # Summary table
    df = build_summary_table(results)
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save summary
    summary_path = output_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Save metadata
    meta = {
        "config": str(config_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "n_results": len(results),
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Generate figures
    if not args.skip_plots:
        print("\nGenerating figures...")
        fig_dir = output_dir / "figures"
        generate_all_figures(results, output_dir=fig_dir)
        print(f"Figures saved to {fig_dir}")

    print(f"\nAll outputs in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
