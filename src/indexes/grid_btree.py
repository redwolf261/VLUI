"""
Grid + Global B-tree control baseline.

Structure: grid partitioning for space decomposition, but ONE global
sorted B-tree backing store for the actual data. This is the control
structure that has partitioned queries but a global mutable index.

Purpose: isolate whether CRI comes from global tree structure or
from query routing. If this shows CRI > 0, it confirms the global
tree is the cause (not the query path).

Baseline B3 in the paper.
"""

from __future__ import annotations

import bisect
import time

import numpy as np

from src.indexes.base import IndexStats, SpatialIndex
from src.models.spatial import (
    SpatialPoint,
    SpatialRegion,
    build_grid_partition,
)


class GridGlobalBTreeIndex(SpatialIndex):
    """Grid partition with a SINGLE global sorted list (B-tree proxy).

    All points go into one sorted array, regardless of region.
    Queries use grid cells to narrow the search, but the underlying
    storage is globally shared.

    Expected: CRI > 0 because the global sorted structure means
    insertions anywhere affect the position of all elements.
    """

    def __init__(
        self,
        dim: int = 2,
        space_lo: float = 0.0,
        space_hi: float = 1.0,
        grid_dims: tuple[int, ...] | None = None,
    ):
        self._dim = dim
        self._space_lo = np.full(dim, space_lo)
        self._space_hi = np.full(dim, space_hi)
        self._grid_dims = grid_dims or tuple([4] * dim)

        self._regions = build_grid_partition(
            self._space_lo, self._space_hi, self._grid_dims
        )

        # SINGLE global sorted list — shared mutable structure
        self._global_data: list[tuple[float, SpatialPoint]] = []
        self._keys: list[float] = []  # parallel key list for O(log n) bisect
        self._total_size = 0

    @property
    def name(self) -> str:
        return f"Grid+GlobalBTree ({'x'.join(map(str, self._grid_dims))})"

    def size(self) -> int:
        return self._total_size

    def _key_from_coords(self, coords) -> float:
        normalized = (coords - self._space_lo) / (self._space_hi - self._space_lo)
        key = 0.0
        for d in range(self._dim):
            key += float(normalized[d]) * (10 ** (self._dim - d - 1))
        return key

    def _sort_key(self, point: SpatialPoint) -> float:
        return self._key_from_coords(point.coords)

    def insert(self, point: SpatialPoint) -> IndexStats:
        key = self._sort_key(point)

        t0 = time.perf_counter_ns()
        # Insert into GLOBAL list — this touches the shared structure
        idx = bisect.bisect_left(self._keys, key)
        self._keys.insert(idx, key)
        self._global_data.insert(idx, (key, point))
        elapsed = time.perf_counter_ns() - t0

        self._total_size += 1

        return IndexStats(
            latency_ns=float(elapsed),
            node_accesses=1,
            structural_mutations=1,
        )

    def delete(self, point: SpatialPoint) -> IndexStats:
        t0 = time.perf_counter_ns()
        for i, (_k, p) in enumerate(self._global_data):
            if p.point_id == point.point_id:
                self._global_data.pop(i)
                self._keys.pop(i)
                self._total_size -= 1
                break
        elapsed = time.perf_counter_ns() - t0
        return IndexStats(latency_ns=float(elapsed), structural_mutations=1)

    def range_query(self, region: SpatialRegion) -> tuple[list[SpatialPoint], IndexStats]:
        """Range query — bisect-narrowed scan of the GLOBAL sorted list.

        The decimal sort key (x*10 + y for 2D) is monotone over axis-aligned
        rectangles: key(lo) <= key(any point in region) <= key(hi). So bisect
        gives a correct candidate window with no false negatives. False positives
        (points in the key range but outside the spatial region) are filtered by
        contains_point(). Worst-case O(n) for strip queries, O(log n + k) typical.
        """
        results: list[SpatialPoint] = []

        t0 = time.perf_counter_ns()
        key_min = self._key_from_coords(region.lo)
        key_max = self._key_from_coords(region.hi)
        lo_idx = bisect.bisect_left(self._keys, key_min)
        hi_idx = bisect.bisect_right(self._keys, key_max)
        for i in range(lo_idx, hi_idx):
            pt = self._global_data[i][1]
            if region.contains_point(pt):
                results.append(pt)
        elapsed = time.perf_counter_ns() - t0

        return results, IndexStats(
            latency_ns=float(elapsed),
            node_accesses=hi_idx - lo_idx,
        )

    def point_query(self, point: SpatialPoint) -> tuple[SpatialPoint | None, IndexStats]:
        key = self._sort_key(point)
        t0 = time.perf_counter_ns()
        idx = bisect.bisect_left(self._keys, key)
        found = None
        for i in range(max(0, idx - 1), min(len(self._global_data), idx + 2)):
            if self._global_data[i][1].point_id == point.point_id:
                found = self._global_data[i][1]
                break
        elapsed = time.perf_counter_ns() - t0
        return found, IndexStats(latency_ns=float(elapsed), node_accesses=1)
