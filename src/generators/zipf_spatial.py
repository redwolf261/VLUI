"""
Generator A2: Zipf-Distributed Spatial Clusters.

Models the core phenomenon: spatially skewed update patterns.
A small number of regions receive disproportionate update load.

This is the PRIMARY generator for demonstrating CRI.

Distribution model:
- Space is divided into m grid cells (regions)
- Update probability for region R_k ∝ 1/k^α  (Zipf's law)
- α = 0 → uniform; α = 1 → moderate skew; α ≥ 2 → extreme skew
- Within each region, points are uniformly distributed

Key parameter α controls skew intensity:
- α ∈ [0.5, 3.0] is the experimental range
- Each value of α produces a different CRI profile
- This directly tests Theorem 1 (unbounded CRI in global trees)

Queries are distributed uniformly across ALL regions to measure
the interference effect: updates concentrate in some regions,
but we measure query latency everywhere.
"""

from __future__ import annotations

import numpy as np

from src.generators.base import BaseGenerator
from src.models.spatial import (
    OpType,
    Operation,
    SpatialPoint,
    SpatialRegion,
)


class ZipfSpatialGenerator(BaseGenerator):
    """Zipf-skewed spatial update workload."""

    def __init__(
        self,
        *,
        n_points: int = 100_000,
        n_queries: int = 10_000,
        alpha: float = 1.5,
        query_radius: float = 0.05,
        dim: int = 2,
        space_lo: float = 0.0,
        space_hi: float = 1.0,
        grid_dims: tuple[int, ...] | None = None,
        seed: int = 42,
    ):
        """
        Args:
            alpha: Zipf exponent. Higher = more skew.
                   0.0 = uniform, 1.0 = moderate, 2.0+ = extreme
        """
        super().__init__(
            n_points=n_points,
            n_queries=n_queries,
            dim=dim,
            space_lo=space_lo,
            space_hi=space_hi,
            grid_dims=grid_dims,
            seed=seed,
        )
        self.alpha = alpha
        self.query_radius = query_radius

    def params(self):
        p = super().params()
        p.update({
            "alpha": self.alpha,
            "query_radius": self.query_radius,
            "workload_type": "zipf_spatial",
        })
        return p

    def _compute_zipf_weights(self) -> np.ndarray:
        """Compute Zipf probability weights over m regions.

        P(region k) ∝ 1 / (k+1)^α   for k = 0, ..., m-1

        Returns:
            Normalized probability vector of shape (m,)
        """
        m = len(self._regions)
        ranks = np.arange(1, m + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, self.alpha)
        return weights / weights.sum()

    def _sample_point_in_region(self, region: SpatialRegion) -> np.ndarray:
        """Sample a uniformly random point within a region's bounding box."""
        return self.rng.uniform(region.lo, region.hi, size=self.dim)

    def _generate(self) -> None:
        m = len(self._regions)
        zipf_weights = self._compute_zipf_weights()

        # Pre-assign each insert to a region via Zipf
        region_assignments = self.rng.choice(m, size=self.n_points, p=zipf_weights)

        total_ops = self.n_points + self.n_queries
        query_timestamps = set(
            self.rng.choice(total_ops, size=self.n_queries, replace=False)
        )

        point_id = 0
        insert_idx = 0

        for t in range(total_ops):
            if t in query_timestamps:
                # Queries are UNIFORM across regions — this is critical.
                # We need to measure latency in cold regions to detect CRI.
                query_region_id = int(self.rng.integers(0, m))
                region = self._regions[query_region_id]
                center = self._sample_point_in_region(region)

                qr = SpatialRegion(
                    lo=np.maximum(center - self.query_radius, self._lo),
                    hi=np.minimum(center + self.query_radius, self._hi),
                    region_id=query_region_id,
                )
                op = Operation(
                    op_type=OpType.RANGE_QUERY,
                    timestamp=t,
                    query_region=qr,
                    region_id=query_region_id,
                )
            else:
                # Insert into Zipf-assigned region
                rid = int(region_assignments[insert_idx])
                region = self._regions[rid]
                coords = self._sample_point_in_region(region)

                pt = SpatialPoint(coords=coords, point_id=point_id, timestamp=t)
                op = Operation(
                    op_type=OpType.INSERT,
                    timestamp=t,
                    point=pt,
                    region_id=rid,
                )
                point_id += 1
                insert_idx += 1

            self._trace.append(op)
