"""
Generator A4: Adversarial Cluster Burst.

This is the STRONGEST generator — designed to maximize CRI in global trees.
It constructs worst-case update sequences that force maximum structural
propagation across the index.

Attack model:
- All updates concentrate in ONE region (the "attack region")
- Points are tightly clustered to force maximum node splits
- Updates arrive in bursts to overwhelm any buffering
- Queries are issued ONLY in distant "victim regions"

This directly tests Theorem 1:
  sup_{U_i} CRI_{i→j} → ∞ for global balanced trees

Design:
- Phase 1: Warm-up — uniform inserts across all regions
- Phase 2: Attack — burst of tightly packed inserts in attack region
- Phase 3: Measure — queries in victim regions to measure latency impact

The burst intensity and cluster tightness are parameterized to sweep
from benign to pathological.

Parameters:
    attack_region_id: which region receives the burst (default: 0)
    victim_region_ids: which regions are queried for CRI measurement
    burst_size: number of points in each burst
    n_bursts: number of burst phases
    cluster_tightness: σ of Gaussian cluster (smaller = more splits)
    warmup_fraction: fraction of total points used for warmup
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


class AdversarialBurstGenerator(BaseGenerator):
    """Adversarial clustered burst workload — worst-case CRI generator."""

    def __init__(
        self,
        *,
        n_points: int = 100_000,
        n_queries: int = 10_000,
        attack_region_id: int = 0,
        victim_region_ids: list[int] | None = None,
        burst_size: int = 5000,
        n_bursts: int = 10,
        cluster_tightness: float = 0.001,
        warmup_fraction: float = 0.2,
        query_radius: float = 0.05,
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
        self.attack_region_id = attack_region_id
        self.victim_region_ids = victim_region_ids
        self.burst_size = burst_size
        self.n_bursts = n_bursts
        self.cluster_tightness = cluster_tightness
        self.warmup_fraction = warmup_fraction
        self.query_radius = query_radius

    def params(self):
        p = super().params()
        p.update({
            "attack_region_id": self.attack_region_id,
            "victim_region_ids": self.victim_region_ids,
            "burst_size": self.burst_size,
            "n_bursts": self.n_bursts,
            "cluster_tightness": self.cluster_tightness,
            "warmup_fraction": self.warmup_fraction,
            "query_radius": self.query_radius,
            "workload_type": "adversarial_burst",
        })
        return p

    def _resolve_victim_regions(self) -> list[int]:
        """Determine victim regions — all regions except attack region."""
        if self.victim_region_ids is not None:
            return self.victim_region_ids
        m = len(self._regions)
        return [i for i in range(m) if i != self.attack_region_id]

    def _generate(self) -> None:
        victim_ids = self._resolve_victim_regions()
        attack_region = self._regions[self.attack_region_id]
        attack_center = attack_region.center

        warmup_points = int(self.n_points * self.warmup_fraction)
        burst_total = self.burst_size * self.n_bursts
        remaining_points = max(0, self.n_points - warmup_points - burst_total)

        point_id = 0
        timestamp = 0

        # ── Phase 1: Warmup ──────────────────────────────────────────
        # Uniform inserts to build a baseline index state.
        # Interleave some baseline queries in victim regions.
        warmup_queries = self.n_queries // 3
        warmup_query_interval = max(1, warmup_points // max(warmup_queries, 1))

        for i in range(warmup_points):
            coords = self.rng.uniform(self._lo, self._hi, size=self.dim)
            region_id = self._find_region(coords)
            pt = SpatialPoint(coords=coords, point_id=point_id, timestamp=timestamp)
            self._trace.append(Operation(
                op_type=OpType.INSERT,
                timestamp=timestamp,
                point=pt,
                region_id=region_id,
            ))
            point_id += 1
            timestamp += 1

            # Interleave baseline queries for pre-attack measurement
            if i > 0 and i % warmup_query_interval == 0 and warmup_queries > 0:
                self._emit_victim_query(victim_ids, timestamp)
                timestamp += 1
                warmup_queries -= 1

        # ── Phase 2: Attack Bursts ───────────────────────────────────
        # Tight Gaussian clusters in attack region.
        # After each burst, issue queries in victim regions → measures CRI.
        queries_per_burst = max(1, (self.n_queries - (self.n_queries // 3)) // self.n_bursts)

        for burst_idx in range(self.n_bursts):
            # Burst: tightly packed inserts
            for _ in range(self.burst_size):
                # Gaussian cluster centered at attack region center
                coords = self.rng.normal(
                    loc=attack_center,
                    scale=self.cluster_tightness,
                    size=self.dim,
                )
                # Clamp to attack region bounds
                coords = np.clip(coords, attack_region.lo, attack_region.hi)

                pt = SpatialPoint(coords=coords, point_id=point_id, timestamp=timestamp)
                self._trace.append(Operation(
                    op_type=OpType.INSERT,
                    timestamp=timestamp,
                    point=pt,
                    region_id=self.attack_region_id,
                ))
                point_id += 1
                timestamp += 1

            # Post-burst queries in victim regions
            for _ in range(queries_per_burst):
                self._emit_victim_query(victim_ids, timestamp)
                timestamp += 1

        # ── Phase 3: Remaining background inserts ────────────────────
        for _ in range(remaining_points):
            coords = self.rng.uniform(self._lo, self._hi, size=self.dim)
            region_id = self._find_region(coords)
            pt = SpatialPoint(coords=coords, point_id=point_id, timestamp=timestamp)
            self._trace.append(Operation(
                op_type=OpType.INSERT,
                timestamp=timestamp,
                point=pt,
                region_id=region_id,
            ))
            point_id += 1
            timestamp += 1

    def _emit_victim_query(self, victim_ids: list[int], timestamp: int) -> None:
        """Issue a range query in a randomly chosen victim region."""
        vid = int(self.rng.choice(victim_ids))
        region = self._regions[vid]
        center = self.rng.uniform(region.lo, region.hi, size=self.dim)
        qr = SpatialRegion(
            lo=np.maximum(center - self.query_radius, region.lo),
            hi=np.minimum(center + self.query_radius, region.hi),
            region_id=vid,
        )
        self._trace.append(Operation(
            op_type=OpType.RANGE_QUERY,
            timestamp=timestamp,
            query_region=qr,
            region_id=vid,
        ))
