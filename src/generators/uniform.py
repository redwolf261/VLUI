"""
Generator A1: Uniform Random Distribution.

Control baseline — no spatial skew, no temporal skew.
Updates and queries are uniformly distributed across the space.

Purpose:
- Establish CRI baseline (expected ≈ 0 even in global trees under uniform load)
- Validate that measurement framework detects no false positives
- Provide comparison anchor for skewed workloads

Parameters:
- n_points: total insert operations
- n_queries: total range queries (interleaved)
- query_radius: half-width of square range queries
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


class UniformRandomGenerator(BaseGenerator):
    """Uniform random spatial workload — the null hypothesis generator."""

    def __init__(
        self,
        *,
        n_points: int = 100_000,
        n_queries: int = 10_000,
        query_radius: float = 0.05,
        query_interleave_ratio: float = 0.1,
        dim: int = 2,
        space_lo: float = 0.0,
        space_hi: float = 1.0,
        grid_dims: tuple[int, ...] | None = None,
        seed: int = 42,
    ):
        super().__init__(
            n_points=n_points,
            n_queries=n_queries,
            dim=dim,
            space_lo=space_lo,
            space_hi=space_hi,
            grid_dims=grid_dims,
            seed=seed,
        )
        self.query_radius = query_radius
        self.query_interleave_ratio = query_interleave_ratio

    def params(self):
        p = super().params()
        p.update({
            "query_radius": self.query_radius,
            "query_interleave_ratio": self.query_interleave_ratio,
        })
        return p

    def _generate(self) -> None:
        total_ops = self.n_points + self.n_queries
        # Decide which timesteps are queries
        query_timestamps = set(
            self.rng.choice(total_ops, size=self.n_queries, replace=False)
        )

        point_id = 0
        query_id = 0
        for t in range(total_ops):
            if t in query_timestamps:
                # Generate uniform random range query
                center = self.rng.uniform(
                    self._lo + self.query_radius,
                    self._hi - self.query_radius,
                    size=self.dim,
                )
                qr = SpatialRegion(
                    lo=center - self.query_radius,
                    hi=center + self.query_radius,
                    region_id=self._find_region(center),
                )
                op = Operation(
                    op_type=OpType.RANGE_QUERY,
                    timestamp=t,
                    query_region=qr,
                    region_id=qr.region_id,
                )
                query_id += 1
            else:
                # Generate uniform random insert
                coords = self.rng.uniform(self._lo, self._hi, size=self.dim)
                pt = SpatialPoint(coords=coords, point_id=point_id, timestamp=t)
                region_id = self._find_region(coords)
                op = Operation(
                    op_type=OpType.INSERT,
                    timestamp=t,
                    point=pt,
                    region_id=region_id,
                )
                point_id += 1

            self._trace.append(op)
