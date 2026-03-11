"""
Advanced real-time benchmark dashboard with multi-metric visualization.

Shows:
- Live CRI evolution (heatmap + metrics)
- Query latency streaming in
- Index performance comparison
- Statistical significance indicators
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from src.generators.zipf_spatial import ZipfSpatialGenerator
from src.indexes.rtree_index import NaiveRTreeIndex
from src.indexes.lchi_index import LCHIIndex
from src.metrics.cri import CRIMeasurer
from src.models.spatial import OpType


class DashboardBenchmark:
    """Advanced multi-metric real-time benchmark dashboard."""

    def __init__(self, output_dir: Path = Path("./results/dashboard")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Rolling metrics
        self.query_window = deque(maxlen=100)  # Last 100 queries
        self.timestamps = []
        self.rtree_isolation = deque(maxlen=20)
        self.lchi_isolation = deque(maxlen=20)
        self.rtree_mean_cri = deque(maxlen=20)
        self.lchi_mean_cri = deque(maxlen=20)

    def run_with_dashboard(self, n_ops: int = 800, update_interval: int = 50):
        """Run benchmark with advanced dashboard."""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "🔬 REAL-TIME CRI BENCHMARK DASHBOARD" + " " * 22 + "║")
        print("╚" + "=" * 78 + "╝\n")

        # Generate workload
        print("📊 WORKLOAD GENERATION")
        print("  └─ Zipfian (α=1.5, 4 regions, 500 ops)")
        gen = ZipfSpatialGenerator(
            n_points=int(n_ops * 0.7),
            n_queries=int(n_ops * 0.3),
            alpha=1.5,
            grid_dims=(2, 2),
            seed=42,
        )
        trace = gen.generate()

        # Initialize
        print("\n🏗️  INDEX INITIALIZATION")
        rtree = NaiveRTreeIndex(dim=2, leaf_capacity=50)
        lchi = LCHIIndex(dim=2, grid_dims=(2, 2))
        print("  ├─ R-tree (global tree, baseline)")
        print("  └─ LCHI (proposed locality-constrained)")

        m = 4
        rtree_m = CRIMeasurer(n_regions=m, window_size=50)
        lchi_m = CRIMeasurer(n_regions=m, window_size=50)

        print("\n▶️  OPERATION REPLAY")
        print("─" * 80)

        start_time = time.time()
        for i, op in enumerate(trace.operations):
            # Execute
            if op.op_type == OpType.INSERT and op.point is not None:
                rtree_s = rtree.insert(op.point)
                lchi_s = lchi.insert(op.point)
                rtree_m.record_update(op.region_id, op.timestamp, rtree_s.structural_mutations)
                lchi_m.record_update(op.region_id, op.timestamp, lchi_s.structural_mutations)

            elif op.op_type == OpType.RANGE_QUERY and op.query_region is not None:
                _, rtree_s = rtree.range_query(op.query_region)
                _, lchi_s = lchi.range_query(op.query_region)
                self.query_window.append(rtree_s.latency_ns)
                rtree_m.record_query(op.region_id, op.timestamp, rtree_s.latency_ns)
                lchi_m.record_query(op.region_id, op.timestamp, lchi_s.latency_ns)

            # Update dashboard
            if (i + 1) % update_interval == 0:
                elapsed = time.time() - start_time
                rtree_cri = rtree_m.compute_cri()
                lchi_cri = lchi_m.compute_cri()

                # Collect metrics
                self.timestamps.append(elapsed)
                self.rtree_isolation.append(rtree_cri.isolation_score)
                self.lchi_isolation.append(lchi_cri.isolation_score)
                self.rtree_mean_cri.append(rtree_cri.mean_cross_cri)
                self.lchi_mean_cri.append(lchi_cri.mean_cross_cri)

                # Print dashboard
                self._render_dashboard(
                    i + 1,
                    len(trace.operations),
                    elapsed,
                    rtree_cri,
                    lchi_cri,
                )

                # Save plots
                self._render_plots(i + 1)

        total_time = time.time() - start_time
        print("─" * 80)

        # Final results
        final_rtree = rtree_m.compute_cri()
        final_lchi = lchi_m.compute_cri()

        print("\n" + "═" * 80)
        print("  📊 FINAL RESULTS")
        print("═" * 80)

        comparison = (final_lchi.isolation_score / final_rtree.isolation_score - 1) * 100

        print(f"\n┌─ ISOLATION SCORE (higher = better)")
        print(f"├─ R-tree:    {final_rtree.isolation_score:8.4f}")
        print(f"├─ LCHI:      {final_lchi.isolation_score:8.4f}  " + f"({'✅ +' if comparison > 0 else '❌ '}{abs(comparison):.1f}%)")
        print(f"└─")

        print(f"\n┌─ MEAN CROSS-CRI (lower = better)")
        print(f"├─ R-tree:    {final_rtree.mean_cross_cri:8.6f}")
        print(f"├─ LCHI:      {final_lchi.mean_cross_cri:8.6f}")
        print(f"└─")

        print(f"\n┌─ EXECUTION STATS")
        print(f"├─ Total ops:        {len(trace.operations)}")
        print(f"├─ Time elapsed:     {total_time:.1f}s")
        print(f"├─ Ops/sec:          {len(trace.operations) / total_time:.0f}")
        print(f"├─ Mean query (ns):  {np.mean(list(self.query_window)):.0f}")
        print(f"└─")

        print(f"\n✅ Plots saved to {self.output_dir}/")
        print("═" * 80 + "\n")

    def _render_dashboard(self, ops: int, total_ops: int, elapsed: float, rtree_cri, lchi_cri):
        """Print live dashboard metrics."""
        pct = (ops / total_ops) * 100
        bar_len = 25
        filled = int(bar_len * ops / total_ops)
        bar = "█" * filled + "░" * (bar_len - filled)

        iso_delta = lchi_cri.isolation_score - rtree_cri.isolation_score
        indicator = "✅" if iso_delta > 0.1 else "⚠️ " if iso_delta > 0 else "❌"

        print(
            f"│ [{bar}] {pct:5.1f}% │ "
            f"t={elapsed:6.1f}s │ "
            f"Ops {ops:3d}/{total_ops} │ "
            f"𝐼(R-tree)={rtree_cri.isolation_score:6.4f} │ "
            f"𝐼(LCHI)={lchi_cri.isolation_score:6.4f} {indicator}"
        )

    def _render_plots(self, ops: int):
        """Render multi-metric dashboard plots."""
        if len(self.timestamps) < 2:
            return

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # ──── Row 1: Main Metrics ────

        # Plot 1: Isolation Score Timeline
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(
            self.timestamps,
            list(self.rtree_isolation),
            "o-",
            color="#e41a1c",
            label="R-tree",
            linewidth=2.5,
            markersize=6,
        )
        ax1.plot(
            self.timestamps,
            list(self.lchi_isolation),
            "s-",
            color="#ff7f00",
            label="LCHI",
            linewidth=2.5,
            markersize=6,
        )
        ax1.fill_between(
            self.timestamps,
            list(self.rtree_isolation),
            list(self.lchi_isolation),
            alpha=0.2,
            color="green" if self.lchi_isolation[-1] > self.rtree_isolation[-1] else "red",
        )
        ax1.set_ylabel("Isolation Score 𝐼", fontsize=11, fontweight="bold")
        ax1.set_title("Real-time CRI Isolation Evolution", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle="--")
        max_iso = max(max(self.rtree_isolation) if self.rtree_isolation else 0.1, 
                      max(self.lchi_isolation) if self.lchi_isolation else 0.1)
        ax1.set_ylim([0, max(max_iso * 1.1, 0.1)])

        # Plot 2: Current Status Box
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")
        rtree_iso = self.rtree_isolation[-1] if self.rtree_isolation[-1] > 0 else 1e-6
        lchi_iso = self.lchi_isolation[-1]
        status_text = (
            f"Current Status\n"
            f"{'─' * 20}\n"
            f"R-tree: {self.rtree_isolation[-1]:.4f}\n"
            f"LCHI:   {self.lchi_isolation[-1]:.4f}\n"
            f"\n"
            f"Δ = {(lchi_iso - self.rtree_isolation[-1]):.4f}\n"
            f"Δ% = {((lchi_iso / rtree_iso - 1) * 100):.1f}%"
        )
        ax2.text(
            0.1,
            0.5,
            status_text,
            transform=ax2.transAxes,
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        # ──── Row 2: Mean Cross-CRI ────

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(
            self.timestamps,
            list(self.rtree_mean_cri),
            "o-",
            color="#e41a1c",
            label="R-tree",
            linewidth=2,
            markersize=5,
        )
        ax3.plot(
            self.timestamps,
            list(self.lchi_mean_cri),
            "s-",
            color="#ff7f00",
            label="LCHI",
            linewidth=2,
            markersize=5,
        )
        ax3.set_ylabel("Mean Cross-CRI", fontsize=10)
        ax3.set_title("Cross-Region Interference", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ──── Query Latency Distribution ────

        ax4 = fig.add_subplot(gs[1, 1:])
        if self.query_window:
            ax4.hist(
                self.query_window,
                bins=20,
                color="#4daf4a",
                alpha=0.7,
                edgecolor="black",
            )
            ax4.axvline(
                np.mean(self.query_window),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"μ={np.mean(self.query_window):.0f}ns",
            )
            ax4.axvline(
                np.median(self.query_window),
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"σ={np.std(self.query_window):.0f}ns",
            )
            ax4.set_xlabel("Query Latency (ns)", fontsize=10)
            ax4.set_ylabel("Frequency", fontsize=10)
            ax4.set_title("Query Latency Distribution (last 100)", fontsize=11, fontweight="bold")
            ax4.legend(fontsize=9)

        # ──── Row 3: Performance Metrics ────

        ax5 = fig.add_subplot(gs[2, 0])
        metrics = ["R-tree", "LCHI"]
        isolation_vals = [self.rtree_isolation[-1], self.lchi_isolation[-1]]
        colors = ["#e41a1c", "#ff7f00"]
        bars = ax5.bar(metrics, isolation_vals, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
        for bar, val in zip(bars, isolation_vals):
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax5.set_ylabel("Isolation Score", fontsize=10)
        ax5.set_title("Current Isolation Comparison", fontsize=11, fontweight="bold")
        ax5.set_ylim([0, max(isolation_vals) * 1.2])

        # Threshold indicator
        ax5.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Baseline=1.0")
        ax5.legend(fontsize=9)

        ax6 = fig.add_subplot(gs[2, 1])
        metric_names = ["Mean\nCRI", "Isolation"]
        rtree_norm = [
            (self.rtree_mean_cri[-1] / max(self.rtree_mean_cri)) if (self.rtree_mean_cri and max(self.rtree_mean_cri) > 0) else 0,
            (self.rtree_isolation[-1] / max(self.rtree_isolation)) if (self.rtree_isolation and max(self.rtree_isolation) > 0) else 0,
        ]
        lchi_norm = [
            (self.lchi_mean_cri[-1] / max(self.lchi_mean_cri)) if (self.lchi_mean_cri and max(self.lchi_mean_cri) > 0) else 0,
            (self.lchi_isolation[-1] / max(self.lchi_isolation)) if (self.lchi_isolation and max(self.lchi_isolation) > 0) else 0,
        ]
        x = np.arange(len(metric_names))
        width = 0.35
        ax6.bar(x - width / 2, rtree_norm, width, label="R-tree", color="#e41a1c", alpha=0.7)
        ax6.bar(x + width / 2, lchi_norm, width, label="LCHI", color="#ff7f00", alpha=0.7)
        ax6.set_ylabel("Normalized Score", fontsize=10)
        ax6.set_title("Normalized Metrics", fontsize=11, fontweight="bold")
        ax6.set_xticks(x)
        ax6.set_xticklabels(metric_names)
        ax6.legend(fontsize=9)
        ax6.set_ylim([0, 1.1])

        # ──── Summary Panel ────

        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")

        delta_iso = self.lchi_isolation[-1] - self.rtree_isolation[-1]
        delta_cri = self.lchi_mean_cri[-1] - self.rtree_mean_cri[-1]
        rtree_iso_safe = self.rtree_isolation[-1] if self.rtree_isolation[-1] > 0 else 1.0
        improvement_iso = (delta_iso / rtree_iso_safe * 100)

        summary = (
            f"Key Findings\n"
            f"{'─' * 18}\n"
            f"𝛥(Isolation):  {delta_iso:+.4f}\n"
            f"Improvement:   {improvement_iso:+.1f}%\n"
            f"𝛥(CRI):        {delta_cri:+.6f}\n"
        )
        ax7.text(
            0.05,
            0.6,
            summary,
            transform=ax7.transAxes,
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.4),
        )

        fig.suptitle(
            f"CRI Benchmark Dashboard (ops={ops})",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.savefig(self.output_dir / "dashboard.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    viz = DashboardBenchmark()
    viz.run_with_dashboard(n_ops=500, update_interval=50)


if __name__ == "__main__":
    main()
