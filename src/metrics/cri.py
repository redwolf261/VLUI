"""
CRI (Cross-Region Interference) Measurement Framework.

This module implements the formal CRI metric from the paper:

    CRI_{i→j} = ∂Q_j / ∂U_i    for i ≠ j

Where:
    Q_j = expected query latency in region R_j
    U_i = update rate in region R_i

Since we measure empirically (not analytically), we estimate CRI via
finite differences over windowed observations.

Measurement Protocol:
1. Divide the trace into time windows of width W
2. In each window w, count:
   - U_i(w): number of updates in region i
   - Q_j(w): average query latency in region j
3. Estimate CRI_{i→j} via regression or finite difference:
   CRI_{i→j} ≈ Cov(U_i, Q_j) / Var(U_i)

This is the empirical operationalization of Definition 1.

Outputs:
- CRI matrix: m×m matrix where entry (i,j) = CRI_{i→j}
- Isolation score: max |CRI_{i→j}| for i≠j
- CRI time series: per-window CRI estimates
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class LatencyRecord:
    """A single query latency observation.

    Attributes:
        region_id: which region the query targeted
        timestamp: logical time of the query
        latency_ns: measured latency in nanoseconds
        node_accesses: number of index nodes accessed (structural cost)
    """
    region_id: int
    timestamp: int
    latency_ns: float
    node_accesses: int = 0


@dataclass
class UpdateRecord:
    """A single update event observation.

    Attributes:
        region_id: which region received the update
        timestamp: logical time of the update
        structural_mutations: number of node splits/merges triggered
    """
    region_id: int
    timestamp: int
    structural_mutations: int = 0


@dataclass
class CRIResult:
    """Complete CRI measurement result.

    Attributes:
        cri_matrix: m×m matrix, entry (i,j) = raw CRI_{i→j} (ns per update)
        cri_normalized: m×m matrix, entry (i,j) = elasticity CRI^norm_{i→j}
        isolation_score: max |CRI^norm_{i→j}| for i≠j
        isolation_score_raw: max |CRI_{i→j}| for i≠j (unnormalized)
        cri_diagonal: self-interference (diagonal entries, normalized)
        n_regions: number of regions
        n_windows: number of time windows used
        window_size: width of each window
        confidence_intervals: optional 95% CI for each CRI estimate
        p_values: optional p-value matrix for statistical significance
    """
    cri_matrix: NDArray[np.float64]         # raw
    cri_normalized: NDArray[np.float64]     # elasticity (dimensionless)
    isolation_score: float                   # max off-diag |CRI^norm|
    isolation_score_raw: float               # max off-diag |CRI_raw|
    cri_diagonal: NDArray[np.float64]
    n_regions: int
    n_windows: int
    window_size: int
    confidence_intervals: NDArray[np.float64] | None = None
    p_values: NDArray[np.float64] | None = None

    @property
    def max_cross_cri(self) -> float:
        """Maximum absolute off-diagonal normalized CRI."""
        return self.isolation_score

    @property
    def mean_cross_cri(self) -> float:
        """Mean absolute off-diagonal normalized CRI."""
        m = self.n_regions
        mask = ~np.eye(m, dtype=bool)
        return float(np.mean(np.abs(self.cri_normalized[mask])))

    @property
    def mean_cross_cri_raw(self) -> float:
        """Mean absolute off-diagonal raw CRI."""
        m = self.n_regions
        mask = ~np.eye(m, dtype=bool)
        return float(np.mean(np.abs(self.cri_matrix[mask])))

    def is_epsilon_isolated(self, epsilon: float) -> bool:
        """Check if the index satisfies ε-isolation (Definition 2).
        Uses normalized CRI for scale-invariant comparison."""
        return self.isolation_score <= epsilon

    def significant_pairs(self, alpha: float = 0.05) -> list[tuple[int, int, float]]:
        """Return (i, j, CRI^norm) pairs with p < alpha."""
        if self.p_values is None:
            return []
        pairs = []
        m = self.n_regions
        for i in range(m):
            for j in range(m):
                if i != j and self.p_values[i, j] < alpha:
                    pairs.append((i, j, float(self.cri_normalized[i, j])))
        return pairs

    def summary(self) -> dict:
        return {
            "isolation_score": self.isolation_score,
            "isolation_score_raw": self.isolation_score_raw,
            "mean_cross_cri": self.mean_cross_cri,
            "mean_cross_cri_raw": self.mean_cross_cri_raw,
            "max_self_cri": float(np.max(np.abs(self.cri_diagonal))),
            "n_regions": self.n_regions,
            "n_windows": self.n_windows,
            "n_significant_pairs_005": len(self.significant_pairs(0.05)),
        }


class CRIMeasurer:
    """Empirical CRI estimator using windowed regression.

    Usage:
        measurer = CRIMeasurer(n_regions=16, window_size=1000)

        # During benchmark execution:
        measurer.record_update(region_id=3, timestamp=42, mutations=2)
        measurer.record_query(region_id=7, timestamp=43, latency_ns=1500)

        # After execution:
        result = measurer.compute_cri()
    """

    def __init__(self, n_regions: int, window_size: int = 1000):
        """
        Args:
            n_regions: number of partition regions (m)
            window_size: number of operations per time window (W)
        """
        self.n_regions = n_regions
        self.window_size = window_size
        self._updates: list[UpdateRecord] = []
        self._queries: list[LatencyRecord] = []

    def record_update(
        self,
        region_id: int,
        timestamp: int,
        structural_mutations: int = 0,
    ) -> None:
        self._updates.append(UpdateRecord(
            region_id=region_id,
            timestamp=timestamp,
            structural_mutations=structural_mutations,
        ))

    def record_query(
        self,
        region_id: int,
        timestamp: int,
        latency_ns: float,
        node_accesses: int = 0,
    ) -> None:
        self._queries.append(LatencyRecord(
            region_id=region_id,
            timestamp=timestamp,
            latency_ns=latency_ns,
            node_accesses=node_accesses,
        ))

    def _build_window_aggregates(self) -> tuple[
        NDArray[np.float64], NDArray[np.float64], int, int, int
    ]:
        """Build per-window update counts and mean latencies.

        Returns:
            (U, Q, n_windows, t_min, t_max) where
            U[w, i] = update count in window w for region i
            Q[w, j] = mean query latency in window w for region j (NaN if no data)
        """
        all_ts = ([u.timestamp for u in self._updates]
                  + [q.timestamp for q in self._queries])
        t_min, t_max = min(all_ts), max(all_ts)
        n_windows = max(1, (t_max - t_min + 1) // self.window_size)

        U = np.zeros((n_windows, self.n_regions), dtype=np.float64)
        Q = np.full((n_windows, self.n_regions), np.nan, dtype=np.float64)
        Q_counts = np.zeros((n_windows, self.n_regions), dtype=np.int64)
        Q_sums = np.zeros((n_windows, self.n_regions), dtype=np.float64)

        for rec in self._updates:
            w = min((rec.timestamp - t_min) // self.window_size, n_windows - 1)
            if 0 <= rec.region_id < self.n_regions:
                U[w, rec.region_id] += 1

        for rec in self._queries:
            w = min((rec.timestamp - t_min) // self.window_size, n_windows - 1)
            if 0 <= rec.region_id < self.n_regions:
                Q_sums[w, rec.region_id] += rec.latency_ns
                Q_counts[w, rec.region_id] += 1

        valid_mask = Q_counts > 0
        Q[valid_mask] = Q_sums[valid_mask] / Q_counts[valid_mask]

        return U, Q, n_windows, t_min, t_max

    def compute_cri(self) -> CRIResult:
        """Compute the full CRI matrix from recorded observations.

        Produces TWO matrices:
        1. Raw CRI:        CRI_{i→j} = Cov(U_i, Q_j) / Var(U_i)  [ns/update]
        2. Normalized CRI: CRI^norm_{i→j} = (∂Q_j/Q̄_j) / (∂U_i/Ū_i)
                          = elasticity, dimensionless, comparable across indexes

        The normalized form is equation (3) in the paper:
            CRI^norm_{i→j} = CRI_{i→j} · (Ū_i / Q̄_j)

        Returns:
            CRIResult with raw matrix, normalized matrix, and diagnostics.
        """
        if not self._updates or not self._queries:
            return self._empty_result()

        U, Q, n_windows, t_min, t_max = self._build_window_aggregates()

        # Compute raw CRI matrix via bivariate regression
        cri_raw = np.zeros((self.n_regions, self.n_regions), dtype=np.float64)
        ci_matrix = np.zeros((self.n_regions, self.n_regions), dtype=np.float64)
        p_matrix = np.ones((self.n_regions, self.n_regions), dtype=np.float64)

        for i in range(self.n_regions):
            for j in range(self.n_regions):
                cri_val, ci_val, p_val = self._estimate_cri_pair(U[:, i], Q[:, j])
                cri_raw[i, j] = cri_val
                ci_matrix[i, j] = ci_val
                p_matrix[i, j] = p_val

        # Normalize: CRI^norm = CRI_raw · (Ū_i / Q̄_j)
        # Ū_i = mean update rate in region i across windows
        # Q̄_j = mean query latency in region j across windows
        U_means = np.nanmean(U, axis=0)  # shape (m,)
        Q_means = np.nanmean(Q, axis=0)  # shape (m,)

        cri_norm = np.zeros_like(cri_raw)
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if abs(Q_means[j]) > 1e-12 and not np.isnan(Q_means[j]):
                    cri_norm[i, j] = cri_raw[i, j] * (U_means[i] / Q_means[j])
                else:
                    cri_norm[i, j] = 0.0

        # Extract scores from NORMALIZED matrix
        diag = np.diag(cri_norm)
        off_diag_mask = ~np.eye(self.n_regions, dtype=bool)
        off_diag_norm = np.abs(cri_norm[off_diag_mask])
        off_diag_raw = np.abs(cri_raw[off_diag_mask])

        isolation_score = float(np.max(off_diag_norm)) if off_diag_norm.size > 0 else 0.0
        isolation_raw = float(np.max(off_diag_raw)) if off_diag_raw.size > 0 else 0.0

        return CRIResult(
            cri_matrix=cri_raw,
            cri_normalized=cri_norm,
            isolation_score=isolation_score,
            isolation_score_raw=isolation_raw,
            cri_diagonal=diag,
            n_regions=self.n_regions,
            n_windows=n_windows,
            window_size=self.window_size,
            confidence_intervals=ci_matrix,
            p_values=p_matrix,
        )

    def _estimate_cri_pair(
        self,
        u_series: NDArray[np.float64],
        q_series: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        """Estimate CRI_{i→j} from per-window U_i and Q_j series.

        Uses: CRI = Cov(U, Q) / Var(U)  (OLS slope)
        With bootstrap confidence interval and permutation p-value.

        Args:
            u_series: update counts per window, shape (W,)
            q_series: mean query latencies per window, shape (W,) with NaNs

        Returns:
            (cri_estimate, confidence_interval_half_width, p_value)
        """
        # Filter windows where both U and Q are valid
        valid = ~np.isnan(q_series) & (u_series >= 0)
        u = u_series[valid]
        q = q_series[valid]

        if len(u) < 3:
            return 0.0, float("inf"), 1.0

        var_u = np.var(u, ddof=1)
        if var_u < 1e-12:
            return 0.0, float("inf"), 1.0

        cov_uq = np.cov(u, q, ddof=1)[0, 1]
        cri = cov_uq / var_u

        rng = np.random.default_rng(hash((u.sum(), q.sum())) % (2**31))
        n = len(u)

        # Bootstrap 95% CI
        n_boot = 200
        boot_cris = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            u_b, q_b = u[idx], q[idx]
            var_b = np.var(u_b, ddof=1)
            if var_b < 1e-12:
                boot_cris[b] = 0.0
            else:
                boot_cris[b] = np.cov(u_b, q_b, ddof=1)[0, 1] / var_b

        ci = float(np.percentile(np.abs(boot_cris - cri), 95))

        # Permutation test for statistical significance
        # H0: no relationship between U_i and Q_j
        n_perm = 200
        abs_cri = abs(cri)
        count_extreme = 0
        for _ in range(n_perm):
            u_perm = rng.permutation(u)
            var_p = np.var(u_perm, ddof=1)
            if var_p < 1e-12:
                continue
            cri_perm = np.cov(u_perm, q, ddof=1)[0, 1] / var_p
            if abs(cri_perm) >= abs_cri:
                count_extreme += 1
        p_value = (count_extreme + 1) / (n_perm + 1)  # +1 for continuity

        return float(cri), ci, p_value

    def _empty_result(self) -> CRIResult:
        m = self.n_regions
        return CRIResult(
            cri_matrix=np.zeros((m, m)),
            cri_normalized=np.zeros((m, m)),
            isolation_score=0.0,
            isolation_score_raw=0.0,
            cri_diagonal=np.zeros(m),
            n_regions=m,
            n_windows=0,
            window_size=self.window_size,
        )

    def compute_structural_cri(self) -> CRIResult:
        """Alternative CRI using structural mutations instead of latency.

        CRI_{i→j}^struct = Cov(mutations_i, Q_j) / Var(mutations_i)

        This isolates structural propagation from other latency factors.
        """
        if not self._updates or not self._queries:
            return self._empty_result()

        all_ts = [u.timestamp for u in self._updates] + [q.timestamp for q in self._queries]
        t_min, t_max = min(all_ts), max(all_ts)
        n_windows = max(1, (t_max - t_min + 1) // self.window_size)

        M = np.zeros((n_windows, self.n_regions), dtype=np.float64)
        Q_sums = np.zeros((n_windows, self.n_regions), dtype=np.float64)
        Q_counts = np.zeros((n_windows, self.n_regions), dtype=np.int64)
        Q = np.full((n_windows, self.n_regions), np.nan)

        for rec in self._updates:
            w = min((rec.timestamp - t_min) // self.window_size, n_windows - 1)
            if 0 <= rec.region_id < self.n_regions:
                M[w, rec.region_id] += rec.structural_mutations

        for rec in self._queries:
            w = min((rec.timestamp - t_min) // self.window_size, n_windows - 1)
            if 0 <= rec.region_id < self.n_regions:
                Q_sums[w, rec.region_id] += rec.latency_ns
                Q_counts[w, rec.region_id] += 1

        valid_mask = Q_counts > 0
        Q[valid_mask] = Q_sums[valid_mask] / Q_counts[valid_mask]

        cri_raw = np.zeros((self.n_regions, self.n_regions))
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                cri_raw[i, j], _, _ = self._estimate_cri_pair(M[:, i], Q[:, j])

        # Normalize structural CRI
        M_means = np.nanmean(M, axis=0)
        Q_means = np.nanmean(Q, axis=0)
        cri_norm = np.zeros_like(cri_raw)
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if abs(Q_means[j]) > 1e-12 and not np.isnan(Q_means[j]):
                    cri_norm[i, j] = cri_raw[i, j] * (M_means[i] / Q_means[j])

        diag = np.diag(cri_norm)
        off_diag_mask = ~np.eye(self.n_regions, dtype=bool)
        off_diag_norm = np.abs(cri_norm[off_diag_mask])
        off_diag_raw = np.abs(cri_raw[off_diag_mask])
        isolation_score = float(np.max(off_diag_norm)) if off_diag_norm.size > 0 else 0.0
        isolation_raw = float(np.max(off_diag_raw)) if off_diag_raw.size > 0 else 0.0

        return CRIResult(
            cri_matrix=cri_raw,
            cri_normalized=cri_norm,
            isolation_score=isolation_score,
            isolation_score_raw=isolation_raw,
            cri_diagonal=diag,
            n_regions=self.n_regions,
            n_windows=n_windows,
            window_size=self.window_size,
        )
