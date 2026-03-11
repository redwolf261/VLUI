"""
Controlled CRI Estimation Protocol.

Implements the formal procedure from the paper Section IV.4:

    To estimate CRI_{i→j}:
    1. Fix query workload in region R_j
    2. Gradually increase update rate in region R_i
    3. Measure change in latency slope

    Fit regression: ΔL_j = β · Δλ_i
    β approximates CRI.

This is a CAUSAL estimation — unlike the observational windowed
regression in cri.py, this holds confounders constant by design.

Protocol:
- Phase 0: Warmup (uniform inserts everywhere, build baseline index)
- Phase 1..K: Steps of increasing update rate in attack region
  - At each step k, inject λ_k updates into region R_i
  - Simultaneously query region R_j at fixed rate
  - Record mean query latency L_j(k)
- Fit: L_j = β₀ + CRI · λ + ε

This produces the gold-standard CRI estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from typing import Callable

from src.indexes.base import SpatialIndex
from src.models.spatial import (
    SpatialPoint,
    SpatialRegion,
    build_grid_partition,
)


@dataclass
class ControlledCRIResult:
    """Result of a controlled CRI experiment for one (i, j) pair.

    Attributes:
        source_region: i — the region receiving updates
        target_region: j — the region being queried
        update_rates: λ values tested, shape (K,)
        mean_latencies: L_j at each λ, shape (K,)
        std_latencies: standard deviation at each step, shape (K,)
        cri_estimate: β from OLS fit (raw, ns per update)
        cri_normalized: β normalized by baseline (dimensionless elasticity)
        r_squared: R² of the linear fit
        p_value: p-value of the slope (t-test)
        baseline_latency: L_j at λ = 0 (intercept)
    """
    source_region: int
    target_region: int
    update_rates: NDArray[np.float64]
    mean_latencies: NDArray[np.float64]
    std_latencies: NDArray[np.float64]
    cri_estimate: float         # raw β
    cri_normalized: float       # elasticity
    r_squared: float
    p_value: float
    baseline_latency: float

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


class ControlledCRIExperiment:
    """Execute the controlled CRI estimation protocol.

    This is computationally expensive but produces the strongest evidence.
    Should be run for selected (source, target) pairs — not all m² pairs.
    """

    def __init__(
        self,
        *,
        dim: int = 2,
        space_lo: float = 0.0,
        space_hi: float = 1.0,
        grid_dims: tuple[int, ...] = (4, 4),
        warmup_points: int = 10000,
        n_rate_steps: int = 10,
        updates_per_step: int = 1000,
        queries_per_step: int = 200,
        query_radius: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            warmup_points: uniform inserts before measurement starts
            n_rate_steps: K steps of increasing update rate
            updates_per_step: λ_base — updates per step (multiplied by step index)
            queries_per_step: fixed query count at each step
            query_radius: half-width of range queries
        """
        self.dim = dim
        self.space_lo = np.full(dim, space_lo)
        self.space_hi = np.full(dim, space_hi)
        self.grid_dims = grid_dims
        self.warmup_points = warmup_points
        self.n_rate_steps = n_rate_steps
        self.updates_per_step = updates_per_step
        self.queries_per_step = queries_per_step
        self.query_radius = query_radius
        self.seed = seed

        self._regions = build_grid_partition(
            self.space_lo, self.space_hi, self.grid_dims
        )

    def run_pair(
        self,
        index: SpatialIndex,
        source_region_id: int,
        target_region_id: int,
    ) -> ControlledCRIResult:
        """Run controlled experiment for one (source, target) pair.

        Args:
            index: fresh spatial index instance
            source_region_id: i — region to inject updates
            target_region_id: j — region to measure queries

        Returns:
            ControlledCRIResult with regression-based CRI estimate.
        """
        rng = np.random.default_rng(self.seed)
        source_r = self._regions[source_region_id]
        target_r = self._regions[target_region_id]
        pid = 0

        # ── Phase 0: Warmup ──────────────────────────────────────
        for _ in range(self.warmup_points):
            coords = rng.uniform(self.space_lo, self.space_hi, size=self.dim)
            pt = SpatialPoint(coords=coords, point_id=pid)
            index.insert(pt)
            pid += 1

        # ── Phase 1..K: Controlled rate sweep ────────────────────
        update_rates = np.zeros(self.n_rate_steps, dtype=np.float64)
        mean_latencies = np.zeros(self.n_rate_steps, dtype=np.float64)
        std_latencies = np.zeros(self.n_rate_steps, dtype=np.float64)

        for step in range(self.n_rate_steps):
            # Number of updates this step: 0, λ, 2λ, ..., (K-1)λ
            n_updates = step * self.updates_per_step
            update_rates[step] = n_updates

            # Inject updates into SOURCE region
            for _ in range(n_updates):
                coords = rng.uniform(source_r.lo, source_r.hi, size=self.dim)
                pt = SpatialPoint(coords=coords, point_id=pid)
                index.insert(pt)
                pid += 1

            # Measure query latency in TARGET region
            latencies = []
            for _ in range(self.queries_per_step):
                center = rng.uniform(target_r.lo, target_r.hi, size=self.dim)
                qr = SpatialRegion(
                    lo=np.maximum(center - self.query_radius, target_r.lo),
                    hi=np.minimum(center + self.query_radius, target_r.hi),
                    region_id=target_region_id,
                )
                _, stats = index.range_query(qr)
                latencies.append(float(stats.latency_ns))

            mean_latencies[step] = np.mean(latencies)
            std_latencies[step] = np.std(latencies, ddof=1)

        # ── Fit regression: L_j = β₀ + CRI · λ + ε ──────────────
        cri_raw, intercept, r_sq, p_val = self._fit_ols(
            update_rates, mean_latencies
        )

        # Normalize: elasticity = CRI_raw · (λ_mean / L_mean)
        lambda_mean = np.mean(update_rates[update_rates > 0]) if np.any(update_rates > 0) else 1.0
        l_mean = np.mean(mean_latencies) if np.mean(mean_latencies) > 0 else 1.0
        cri_norm = cri_raw * (lambda_mean / l_mean)

        return ControlledCRIResult(
            source_region=source_region_id,
            target_region=target_region_id,
            update_rates=update_rates,
            mean_latencies=mean_latencies,
            std_latencies=std_latencies,
            cri_estimate=cri_raw,
            cri_normalized=float(cri_norm),
            r_squared=r_sq,
            p_value=p_val,
            baseline_latency=float(intercept),
        )

    def run_full_matrix(
        self,
        index_factory: Callable[[], SpatialIndex],
    ) -> list[ControlledCRIResult]:
        """Run controlled experiment for all (i, j) pairs where i ≠ j.

        Args:
            index_factory: callable that returns a fresh SpatialIndex instance

        Note: This creates a fresh index for each pair to avoid contamination.
        Very expensive — O(m² · K · updates_per_step) total operations.
        """
        m = len(self._regions)
        results = []
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                index = index_factory()
                result = self.run_pair(index, i, j)
                results.append(result)
        return results

    @staticmethod
    def _fit_ols(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[float, float, float, float]:
        """Ordinary least squares: y = β₀ + β₁·x + ε.

        Returns:
            (slope, intercept, r_squared, p_value_for_slope)
        """
        from scipy import stats as sp_stats

        n = len(x)
        if n < 3:
            return 0.0, float(np.mean(y)), 0.0, 1.0

        result = sp_stats.linregress(x, y)  # type: ignore[reportUnknownMemberType]
        return (
            float(result.slope),  # type: ignore[reportUnknownArgumentType]
            float(result.intercept),  # type: ignore[reportUnknownArgumentType]
            float(result.rvalue ** 2),  # type: ignore[reportUnknownArgumentType]
            float(result.pvalue),  # type: ignore[reportUnknownArgumentType]
        )
