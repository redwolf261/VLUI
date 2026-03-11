"""
Publication-quality visualization for CRI experiments.

Generates all key figures for the paper:
1. CRI heatmap matrix           — Figure 3: cross-region interference pattern
2. Isolation score vs α curve   — Figure 4: LCHI vs baselines under Zipf sweep
3. Per-region latency box plots — Figure 5: fairness / tail-latency comparison
4. Controlled CRI regression    — Figure 6: causal dosing experiment
5. Summary comparison table     — Table 1: all (index × workload) results
6. CRI time-series              — Appendix: interference over trace timeline

All figures are saved as both PDF (for LaTeX) and PNG (for README/slides).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from src.benchmark.executor import BenchmarkResult
from src.metrics.cri import CRIResult

# ── Publication style ────────────────────────────────────────────────

STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(STYLE)

# Color palette for indexes
INDEX_COLORS = {
    "NaiveRTree": "#e41a1c",      # red
    "BufferedRTree": "#377eb8",    # blue
    "GridGlobalBTree": "#4daf4a", # green
    "LSMSpatial": "#984ea3",      # purple
    "LCHI": "#ff7f00",             # orange (proposed)
}

INDEX_MARKERS = {
    "NaiveRTree": "o",
    "BufferedRTree": "s",
    "GridGlobalBTree": "^",
    "LSMSpatial": "D",
    "LCHI": "*",
}

def _get_color(name: str) -> str:
    for key, color in INDEX_COLORS.items():
        if key in name:
            return color
    return "#333333"

def _get_marker(name: str) -> str:
    for key, marker in INDEX_MARKERS.items():
        if key in name:
            return marker
    return "o"

def _save(fig: Figure, output_dir: Path, name: str) -> None:
    """Save figure as PDF + PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / f"{name}.pdf"))
    fig.savefig(str(output_dir / f"{name}.png"))
    plt.close(fig)


# ── Figure 1: CRI Heatmap Matrix ────────────────────────────────────

def plot_cri_heatmap(
    cri_result: CRIResult,
    title: str = "CRI Matrix",
    output_dir: Path = Path("figures"),
    filename: str = "cri_heatmap",
    use_normalized: bool = True,
) -> None:
    """Plot m×m CRI matrix as annotated heatmap.

    Diagonal = self-interference, off-diagonal = cross-region interference.
    """
    matrix = cri_result.cri_normalized if use_normalized else cri_result.cri_matrix
    m = cri_result.n_regions

    fig, ax = plt.subplots(figsize=(5, 4))

    # Diverging colormap centered at 0
    vmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    # Annotate each cell
    for i in range(m):
        for j in range(m):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=max(6, 8 - m // 4), color=color)

    # Highlight significant pairs
    if cri_result.p_values is not None:
        for i in range(m):
            for j in range(m):
                if i != j and cri_result.p_values[i, j] < 0.05:
                    rect = Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="gold", linewidth=2,
                    )
                    ax.add_patch(rect)

    ax.set_xlabel("Target region j (query)")
    ax.set_ylabel("Source region i (update)")
    ax.set_title(title)
    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels([f"R{i}" for i in range(m)])
    ax.set_yticklabels([f"R{i}" for i in range(m)])
    fig.colorbar(im, ax=ax, label="CRI" + (" (elasticity)" if use_normalized else " (ns/update)"))

    _save(fig, output_dir, filename)


# ── Figure 2: Isolation Score vs Skew (α sweep) ─────────────────────

def plot_isolation_vs_alpha(
    results: Sequence[BenchmarkResult],
    alphas: Sequence[float] | None = None,
    output_dir: Path = Path("figures"),
    filename: str = "isolation_vs_alpha",
) -> None:
    """Plot isolation score vs Zipf α for all indexes.

    Expects results from zipf workloads with varying α values.
    Groups results by index name and plots one line per index.
    """
    # Group results by index name
    by_index: dict[str, list[tuple[float, float]]] = {}

    for r in results:
        # Extract alpha from workload params or name
        alpha = r.workload_params.get("alpha")
        if alpha is None:
            # Try parsing from workload name like "zipf_alpha_2.0"
            name_parts = r.workload_name.split("_")
            for i, p in enumerate(name_parts):
                if p == "alpha" and i + 1 < len(name_parts):
                    try:
                        alpha = float(name_parts[i + 1])
                    except ValueError:
                        pass
        if alpha is None:
            continue

        idx_name = r.index_name
        if idx_name not in by_index:
            by_index[idx_name] = []
        by_index[idx_name].append((alpha, r.cri_result.isolation_score))

    fig, ax = plt.subplots(figsize=(6, 4))

    for idx_name, points in sorted(by_index.items()):
        points.sort()
        xs, ys = zip(*points)
        ax.plot(
            xs, ys,
            marker=_get_marker(idx_name),
            color=_get_color(idx_name),
            label=idx_name,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Zipf skew parameter α")
    ax.set_ylabel("Isolation score (max |CRI^norm|)")
    ax.set_title("Cross-Region Interference vs. Workload Skew")
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.7, label="ε = 0.01")
    ax.legend()
    ax.set_yscale("log")

    _save(fig, output_dir, filename)


# ── Figure 3: Per-Region Latency Box Plots ──────────────────────────

def plot_region_latency_boxplots(
    results: Sequence[BenchmarkResult],
    workload_filter: str | None = None,
    output_dir: Path = Path("figures"),
    filename: str = "region_latency_boxplots",
) -> None:
    """Box plots of query latency per region, side-by-side for each index.

    Shows whether LCHI achieves fair latency (small spread) while
    baselines show region-dependent latency spikes.
    """
    filtered = results
    if workload_filter:
        filtered = [r for r in results if workload_filter in r.workload_name]

    if not filtered:
        return

    from typing import Any
    n_idx = len(filtered)
    fig, axes_raw = plt.subplots(1, n_idx, figsize=(4 * n_idx, 4), sharey=True)
    axes_list: list[Any] = [axes_raw] if n_idx == 1 else list(axes_raw)  # type: ignore[arg-type]

    for ax, result in zip(axes_list, filtered):
        data: list[Any] = []
        labels: list[str] = []
        for rid in sorted(result.region_query_latencies.keys()):
            lats = result.region_query_latencies[rid]
            if lats:
                data.append(np.array(lats) / 1000)  # convert ns → μs
                labels.append(f"R{rid}")

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
            color = _get_color(result.index_name)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_xlabel("Region")
        ax.set_title(result.index_name, fontsize=9)

    axes_list[0].set_ylabel("Query latency (μs)")
    fig.suptitle(f"Per-Region Query Latency Distribution", fontsize=11)
    fig.tight_layout()

    _save(fig, output_dir, filename)


# ── Figure 4: Controlled CRI Regression Plot ────────────────────────

def plot_controlled_cri(
    rate_steps: Sequence[float],
    measured_latencies: Sequence[float],
    slope: float,
    r_squared: float,
    source_region: int,
    target_region: int,
    output_dir: Path = Path("figures"),
    filename: str = "controlled_cri_regression",
) -> None:
    """Plot controlled (causal) CRI experiment results.

    X-axis: induced update rate in source region
    Y-axis: measured query latency in target region
    + regression line with slope = CRI estimate
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    xs = np.array(rate_steps)
    ys = np.array(measured_latencies)

    ax.scatter(xs, ys, c="#377eb8", s=60, zorder=3, label="Measured")

    # Regression line
    x_fit = np.linspace(xs.min(), xs.max(), 100)
    intercept = np.mean(ys) - slope * np.mean(xs)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, "r--", linewidth=2,
            label=f"OLS: slope={slope:.4f}, R²={r_squared:.3f}")

    ax.set_xlabel(f"Update rate in R{source_region} (ops/window)")
    ax.set_ylabel(f"Query latency in R{target_region} (ns)")
    ax.set_title(f"Controlled CRI: R{source_region} → R{target_region}")
    ax.legend()

    _save(fig, output_dir, filename)


# ── Figure 5: Multi-Index Summary Bar Chart ─────────────────────────

def plot_summary_bars(
    results: Sequence[BenchmarkResult],
    metric: str = "isolation_score",
    output_dir: Path = Path("figures"),
    filename: str = "summary_bars",
) -> None:
    """Grouped bar chart comparing a metric across all (index, workload) pairs."""
    # Group by workload
    workloads = sorted(set(r.workload_name for r in results))
    indexes = sorted(set(r.index_name for r in results))

    data = {}
    for r in results:
        data[(r.workload_name, r.index_name)] = r.summary().get(metric, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(workloads) * 1.5), 5))

    x = np.arange(len(workloads))
    width = 0.8 / len(indexes)

    for k, idx_name in enumerate(indexes):
        vals = [data.get((wl, idx_name), 0) for wl in workloads]
        offset = (k - len(indexes) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=idx_name,
            color=_get_color(idx_name),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Index Comparison: {metric.replace('_', ' ').title()}")
    ax.legend(fontsize=7)
    fig.tight_layout()

    _save(fig, output_dir, filename)


# ── Figure 6: CRI Time Series ───────────────────────────────────────

def plot_cri_timeseries(
    results: Sequence[BenchmarkResult],
    source_region: int,
    target_region: int,
    output_dir: Path = Path("figures"),
    filename: str = "cri_timeseries",
) -> None:
    """Plot per-window CRI_{source→target} over time for each index.

    Useful for showing how moving hotspot dynamically changes interference.
    Requires re-computing windowed CRI (not just the aggregate).
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for result in results:
        # We need the raw measurer data — approximate from region latencies
        # For a full implementation, we'd store per-window CRI in the result
        # For now, plot the per-region query latencies as a rolling mean
        lats = result.region_query_latencies.get(target_region, [])
        if not lats:
            continue

        lats_arr = np.array(lats)
        window = min(50, len(lats_arr) // 4)
        if window < 2:
            continue

        # Rolling mean
        cumsum = np.cumsum(np.insert(lats_arr, 0, 0))
        rolling = (cumsum[window:] - cumsum[:-window]) / window

        ax.plot(
            rolling / 1000,  # ns → μs
            label=result.index_name,
            color=_get_color(result.index_name),
            linewidth=1.0,
            alpha=0.8,
        )

    ax.set_xlabel("Query index (time)")
    ax.set_ylabel(f"Rolling mean latency in R{target_region} (μs)")
    ax.set_title(f"Query Latency Timeline in Region {target_region}")
    ax.legend()

    _save(fig, output_dir, filename)


# ── Table 1: Summary Results DataFrame ──────────────────────────────

def build_summary_table(results: Sequence[BenchmarkResult]) -> pd.DataFrame:
    """Build summary DataFrame from all experiment results.

    Columns:
        Index, Workload, Mean Query (μs), P99 Query (μs),
        Isolation Score, Mean CRI, Sig. Pairs, ε-isolated
    """
    rows = []
    for r in results:
        s = r.summary()
        rows.append({
            "Index": r.index_name,
            "Workload": r.workload_name,
            "Mean Query (μs)": s["mean_query_ns"] / 1000,
            "P50 Query (μs)": s["p50_query_ns"] / 1000,
            "P99 Query (μs)": s["p99_query_ns"] / 1000,
            "Isolation Score": s["isolation_score"],
            "Isolation (raw)": s["isolation_score_raw"],
            "Mean Cross-CRI": s["mean_cross_cri"],
            "Sig. Pairs (p<.05)": s["n_significant_pairs"],
            "ε-iso (0.01)": s["epsilon_isolated_0.01"],
            "ε-iso (0.1)": s["epsilon_isolated_0.1"],
        })
    return pd.DataFrame(rows)


def save_summary_table(
    results: Sequence[BenchmarkResult],
    output_dir: Path = Path("figures"),
    filename: str = "summary_table",
) -> pd.DataFrame:
    """Save summary as CSV + LaTeX table."""
    df = build_summary_table(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / f"{filename}.csv", index=False)

    # LaTeX table
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="CRI comparison across indexes and workloads",
        label="tab:cri_comparison",
    )
    (output_dir / f"{filename}.tex").write_text(latex)

    return df


# ── Master plot function ────────────────────────────────────────────

def generate_all_figures(
    results: Sequence[BenchmarkResult],
    output_dir: Path = Path("figures"),
) -> None:
    """Generate all publication figures from experiment results.

    Call this after running the full experiment suite.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CRI heatmaps for each (index, workload) pair
    for r in results:
        safe = f"{r.workload_name}_{r.index_name}".replace(" ", "_")
        plot_cri_heatmap(
            r.cri_result,
            title=f"CRI: {r.index_name} / {r.workload_name}",
            output_dir=output_dir / "heatmaps",
            filename=safe,
        )

    # 2. Isolation vs alpha (Zipf sweep)
    zipf_results = [r for r in results if "zipf" in r.workload_name]
    if zipf_results:
        plot_isolation_vs_alpha(zipf_results, output_dir=output_dir)

    # 3. Region latency boxplots for key workloads
    for wl_name in ["zipf_alpha_2.0", "adversarial_burst", "hotspot_linear"]:
        wl_results = [r for r in results if r.workload_name == wl_name]
        if wl_results:
            plot_region_latency_boxplots(
                wl_results,
                output_dir=output_dir,
                filename=f"region_latency_{wl_name}",
            )

    # 4. Summary bars
    plot_summary_bars(
        results, metric="isolation_score",
        output_dir=output_dir, filename="isolation_score_comparison",
    )
    plot_summary_bars(
        results, metric="mean_query_ns",
        output_dir=output_dir, filename="query_latency_comparison",
    )

    # 5. Timeseries for moving hotspot
    hotspot_results = [r for r in results if "hotspot" in r.workload_name]
    if hotspot_results:
        plot_cri_timeseries(
            hotspot_results,
            source_region=0,
            target_region=3,
            output_dir=output_dir,
            filename="hotspot_timeseries",
        )

    # 6. Summary table
    save_summary_table(results, output_dir=output_dir)

    print(f"All figures saved to {output_dir}/")
