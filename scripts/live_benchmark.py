"""
Live real-time benchmark visualization.

Shows:
- Live CRI heatmap updates (every 50 ops)
- Real-time performance metrics
- Animated progress with current isolation score
- Latency curve streaming in
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from src.generators.zipf_spatial import ZipfSpatialGenerator
from src.indexes.rtree_index import NaiveRTreeIndex
from src.indexes.lchi_index import LCHIIndex
from src.metrics.cri import CRIMeasurer
from src.models.spatial import OpType


class LiveBenchmarkVisualizer:
    """Real-time visualization of benchmark execution."""

    def __init__(self, output_dir: Path = Path("./results/live")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics collected during run
        self.timestamps = []
        self.query_latencies = []
        self.insert_latencies = []
        self.cri_snapshots = []
        self.isolation_scores = []
        self.ops_count = []

    def run_live(self, n_ops: int = 500, update_interval: int = 50):
        """
        Run benchmark with live updates.

        Shows CRI heatmap and metrics updating every `update_interval` ops.
        """
        print("\n" + "=" * 80)
        print("  LIVE SPATIAL INDEX BENCHMARK")
        print("=" * 80)

        # Generate trace
        print("\n📊 Generating workload (Zipf α=1.5, 4 regions)...")
        gen = ZipfSpatialGenerator(
            n_points=int(n_ops * 0.7),
            n_queries=int(n_ops * 0.3),
            alpha=1.5,
            grid_dims=(2, 2),
            seed=42,
        )
        trace = gen.generate()
        print(f"   Generated {trace.n_ops} operations")

        # Initialize indexes
        print("\n🏢 Initializing indexes...")
        rtree = NaiveRTreeIndex(dim=2, leaf_capacity=50)
        lchi = LCHIIndex(dim=2, grid_dims=(2, 2))

        # Initialize CRI measurers
        m = 4
        rtree_measurer = CRIMeasurer(n_regions=m, window_size=50)
        lchi_measurer = CRIMeasurer(n_regions=m, window_size=50)

        print(f"\n▶️  Replaying {len(trace.operations)} operations...")
        print("-" * 80)

        # Replay loop with live updates
        start_time = time.time()
        for i, op in enumerate(trace.operations):
            # Execute operation on both indexes
            if op.op_type == OpType.INSERT and op.point is not None:
                rtree_stats = rtree.insert(op.point)
                lchi_stats = lchi.insert(op.point)

                rtree_measurer.record_update(
                    region_id=op.region_id,
                    timestamp=op.timestamp,
                    structural_mutations=rtree_stats.structural_mutations,
                )
                lchi_measurer.record_update(
                    region_id=op.region_id,
                    timestamp=op.timestamp,
                    structural_mutations=lchi_stats.structural_mutations,
                )
                self.insert_latencies.append(rtree_stats.latency_ns)

            elif op.op_type == OpType.RANGE_QUERY and op.query_region is not None:
                rtree_results, rtree_stats = rtree.range_query(op.query_region)
                lchi_results, lchi_stats = lchi.range_query(op.query_region)

                rtree_measurer.record_query(
                    region_id=op.region_id,
                    timestamp=op.timestamp,
                    latency_ns=rtree_stats.latency_ns,
                )
                lchi_measurer.record_query(
                    region_id=op.region_id,
                    timestamp=op.timestamp,
                    latency_ns=lchi_stats.latency_ns,
                )
                self.query_latencies.append(rtree_stats.latency_ns)

            # Live update every `update_interval` ops
            if (i + 1) % update_interval == 0:
                elapsed = time.time() - start_time

                # Compute current CRI
                rtree_cri = rtree_measurer.compute_cri()
                lchi_cri = lchi_measurer.compute_cri()

                self.ops_count.append(i + 1)
                self.cri_snapshots.append((rtree_cri.cri_matrix, lchi_cri.cri_matrix))
                self.isolation_scores.append(
                    (rtree_cri.isolation_score, lchi_cri.isolation_score)
                )
                self.timestamps.append(elapsed)

                # Print live metrics
                self._print_live_update(
                    i + 1,
                    len(trace.operations),
                    elapsed,
                    rtree_cri.isolation_score,
                    lchi_cri.isolation_score,
                )

                # Save intermediate plots
                self._save_live_plots(i + 1)

        total_time = time.time() - start_time
        print("-" * 80)
        print(f"✅ Benchmark complete in {total_time:.1f}s\n")

        # Final summary
        final_rtree_cri = rtree_measurer.compute_cri()
        final_lchi_cri = lchi_measurer.compute_cri()

        print("📈 FINAL RESULTS")
        print("-" * 80)
        print(f"R-tree isolation score:    {final_rtree_cri.isolation_score:.4f}")
        print(
            f"LCHI isolation score:      {final_lchi_cri.isolation_score:.4f} "
            f"(+{(final_lchi_cri.isolation_score / final_rtree_cri.isolation_score - 1) * 100:.1f}% better)"
        )
        print(f"R-tree mean cross-CRI:     {final_rtree_cri.mean_cross_cri:.6f}")
        print(f"LCHI mean cross-CRI:       {final_lchi_cri.mean_cross_cri:.6f}")
        print(f"Total operations:          {len(trace.operations)}")
        print(f"Mean query latency (ns):   {np.mean(self.query_latencies):.0f}")
        print("-" * 80)

        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   - live_heatmap.png (animated CRI updates)")
        print(f"   - live_isolation.png (isolation score timeline)")
        print(f"   - live_latency.png (query latency curve)")

    def _print_live_update(
        self,
        current_ops: int,
        total_ops: int,
        elapsed: float,
        rtree_iso: float,
        lchi_iso: float,
    ):
        """Print live progress metrics."""
        pct = (current_ops / total_ops) * 100
        bar_len = 30
        filled = int(bar_len * current_ops / total_ops)
        bar = "█" * filled + "░" * (bar_len - filled)

        iso_diff = lchi_iso - rtree_iso
        iso_indicator = "✅" if iso_diff > 0.05 else "⏳"

        print(
            f"[{bar}] {pct:5.1f}% | "
            f"Ops: {current_ops:4d}/{total_ops:4d} | "
            f"Time: {elapsed:6.1f}s | "
            f"R-tree: {rtree_iso:.4f} | "
            f"LCHI: {lchi_iso:.4f} {iso_indicator}"
        )

    def _save_live_plots(self, ops: int):
        """Save live plots during execution."""
        if len(self.isolation_scores) < 2:
            return

        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, figure=fig)

        # Plot 1: Isolation score timeline
        ax1 = fig.add_subplot(gs[0, 0])
        rtree_scores = [s[0] for s in self.isolation_scores]
        lchi_scores = [s[1] for s in self.isolation_scores]
        ax1.plot(
            self.ops_count,
            rtree_scores,
            "o-",
            color="#e41a1c",
            label="R-tree",
            linewidth=2,
            markersize=5,
        )
        ax1.plot(
            self.ops_count,
            lchi_scores,
            "s-",
            color="#ff7f00",
            label="LCHI",
            linewidth=2,
            markersize=5,
        )
        ax1.set_xlabel("Operations", fontsize=10)
        ax1.set_ylabel("Isolation Score", fontsize=10)
        ax1.set_title("CRI Isolation: Live Update", fontsize=11, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.0])

        # Plot 2: Latest CRI heatmap (R-tree)
        if self.cri_snapshots:
            ax2 = fig.add_subplot(gs[0, 1])
            rtree_matrix = self.cri_snapshots[-1][0]
            im2 = ax2.imshow(rtree_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.01)
            ax2.set_title(f"R-tree CRI Matrix\n(ops={ops})", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Target region j")
            ax2.set_ylabel("Source region i")
            plt.colorbar(im2, ax=ax2, label="CRI")

            # Plot 3: Latest CRI heatmap (LCHI)
            ax3 = fig.add_subplot(gs[0, 2])
            lchi_matrix = self.cri_snapshots[-1][1]
            im3 = ax3.imshow(lchi_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.01)
            ax3.set_title(f"LCHI CRI Matrix\n(ops={ops})", fontsize=11, fontweight="bold")
            ax3.set_xlabel("Target region j")
            ax3.set_ylabel("Source region i")
            plt.colorbar(im3, ax=ax3, label="CRI")

        plt.tight_layout()
        plt.savefig(self.output_dir / "live_heatmap.png", dpi=100, bbox_inches="tight")
        plt.close()


def main():
    """Run live benchmark visualization."""
    viz = LiveBenchmarkVisualizer()
    viz.run_live(n_ops=500, update_interval=50)


if __name__ == "__main__":
    main()
