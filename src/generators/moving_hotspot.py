"""
Generator A3: Moving Hotspot.

Models temporal non-stationarity: the hotspot region changes over time.

This captures real scenarios like:
- Rush hour traffic shifting across city zones
- Event crowds moving between venues
- Sensor deployments migrating across geography

Model:
- A circular/square hotspot of radius r moves through space
- At each timestep, hotspot center advances by velocity v
- fraction β of updates land inside the hotspot
- remaining (1-β) updates are uniform background noise
- Path is parameterized (linear, circular, random walk)

This generator tests whether CRI is transient or cumulative:
- If CRI decays after the hotspot moves away → structure recovers
- If CRI persists → structural damage is permanent (important finding)

Queries are uniform across space to detect interference everywhere.
"""

from __future__ import annotations

import enum

import numpy as np

from src.generators.base import BaseGenerator
from src.models.spatial import (
    OpType,
    Operation,
    SpatialPoint,
    SpatialRegion,
)


class HotspotPath(enum.Enum):
    LINEAR = "linear"       # straight line across space
    CIRCULAR = "circular"   # orbit around center
    RANDOM_WALK = "random_walk"  # Brownian-like motion


class MovingHotspotGenerator(BaseGenerator):
    """Moving spatial hotspot workload generator."""

    def __init__(
        self,
        *,
        n_points: int = 100_000,
        n_queries: int = 10_000,
        hotspot_radius: float = 0.1,
        hotspot_intensity: float = 0.8,
        velocity: float = 0.002,
        path_type: HotspotPath = HotspotPath.LINEAR,
        query_radius: float = 0.05,
        dim: int = 2,
        space_lo: float = 0.0,
        space_hi: float = 1.0,
        grid_dims: tuple[int, ...] | None = None,
        seed: int = 42,
    ):
        """
        Args:
            hotspot_radius: radius of the hotspot region
            hotspot_intensity: β — fraction of updates in hotspot (0.0–1.0)
            velocity: distance hotspot center moves per timestep
            path_type: trajectory shape
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
        self.hotspot_radius = hotspot_radius
        self.hotspot_intensity = hotspot_intensity
        self.velocity = velocity
        self.path_type = path_type
        self.query_radius = query_radius

    def params(self):
        p = super().params()
        p.update({
            "hotspot_radius": self.hotspot_radius,
            "hotspot_intensity": self.hotspot_intensity,
            "velocity": self.velocity,
            "path_type": self.path_type.value,
            "query_radius": self.query_radius,
            "workload_type": "moving_hotspot",
        })
        return p

    def _hotspot_center(self, t: int, total_t: int) -> np.ndarray:
        """Compute hotspot center at logical time t."""
        margin = self.hotspot_radius
        lo = self._lo + margin
        hi = self._hi - margin
        span = hi - lo

        if self.path_type == HotspotPath.LINEAR:
            # Linear sweep from lo to hi
            frac = t / max(total_t - 1, 1)
            return lo + frac * span

        elif self.path_type == HotspotPath.CIRCULAR:
            # Circular orbit around space center
            center = (self._lo + self._hi) / 2.0
            radius = min(span) * 0.35
            angle = 2 * np.pi * t / max(total_t - 1, 1)
            offset = np.zeros(self.dim)
            offset[0] = radius * np.cos(angle)
            offset[1] = radius * np.sin(angle)
            return center + offset

        elif self.path_type == HotspotPath.RANDOM_WALK:
            # Pre-seeded random walk (deterministic given seed)
            rw_rng = np.random.default_rng(self.seed + 9999)
            pos = (self._lo + self._hi) / 2.0  # start at center
            for _ in range(t):
                step = rw_rng.normal(0, self.velocity, size=self.dim)
                pos = np.clip(pos + step, lo, hi)
            return pos

        else:
            raise ValueError(f"Unknown path type: {self.path_type}")

    def _sample_in_hotspot(self, center: np.ndarray) -> np.ndarray:
        """Sample a point uniformly within the hotspot region."""
        offset = self.rng.uniform(
            -self.hotspot_radius, self.hotspot_radius, size=self.dim
        )
        point = center + offset
        return np.clip(point, self._lo, self._hi)

    def _generate(self) -> None:
        total_ops = self.n_points + self.n_queries
        query_timestamps = set(
            self.rng.choice(total_ops, size=self.n_queries, replace=False)
        )

        # Pre-compute hotspot centers for efficiency (avoid O(t^2) random walk)
        if self.path_type == HotspotPath.RANDOM_WALK:
            hotspot_centers = self._precompute_random_walk(total_ops)
        else:
            hotspot_centers = None

        m = len(self._regions)
        point_id = 0

        for t in range(total_ops):
            if hotspot_centers is not None:
                hc = hotspot_centers[t]
            else:
                hc = self._hotspot_center(t, total_ops)

            if t in query_timestamps:
                # Uniform queries across all regions
                query_region_id = int(self.rng.integers(0, m))
                region = self._regions[query_region_id]
                center = self.rng.uniform(region.lo, region.hi, size=self.dim)
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
                # Decide: hotspot or background
                if self.rng.random() < self.hotspot_intensity:
                    coords = self._sample_in_hotspot(hc)
                else:
                    coords = self.rng.uniform(self._lo, self._hi, size=self.dim)

                region_id = self._find_region(coords)
                pt = SpatialPoint(coords=coords, point_id=point_id, timestamp=t)
                op = Operation(
                    op_type=OpType.INSERT,
                    timestamp=t,
                    point=pt,
                    region_id=region_id,
                )
                point_id += 1

            self._trace.append(op)

    def _precompute_random_walk(self, total_ops: int) -> list[np.ndarray]:
        """Pre-compute all random walk positions to avoid O(t^2)."""
        margin = self.hotspot_radius
        lo = self._lo + margin
        hi = self._hi - margin
        rw_rng = np.random.default_rng(self.seed + 9999)
        pos = (self._lo + self._hi) / 2.0
        centers = [pos.copy()]
        for _ in range(1, total_ops):
            step = rw_rng.normal(0, self.velocity, size=self.dim)
            pos = np.clip(pos + step, lo, hi)
            centers.append(pos.copy())
        return centers
