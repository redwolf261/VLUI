"""
Naive in-memory R-tree index wrapper using the `rtree` library.

This serves as the primary BASELINE — a global balanced spatial tree
that is expected to exhibit non-zero CRI under skewed workloads.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.indexes.base import IndexStats, SpatialIndex
from src.models.spatial import SpatialPoint, SpatialRegion


class NaiveRTreeIndex(SpatialIndex):
    """Wrapper around rtree.Index for CRI benchmarking.

    This is a GLOBAL tree — all points share one structure.
    Node splits propagate to the root.
    Expected: CRI > 0 under skew.
    """

    def __init__(self, dim: int = 2, leaf_capacity: int = 50):
        from rtree import index as rtree_index

        p = rtree_index.Property()
        p.dimension = dim
        p.leaf_capacity = leaf_capacity
        p.fill_factor = 0.7

        self._dim = dim
        self._idx = rtree_index.Index(properties=p)
        self._points: dict[int, SpatialPoint] = {}
        self._mutation_counter = 0

    @property
    def name(self) -> str:
        return "R-tree (global)"

    def size(self) -> int:
        return len(self._points)

    def insert(self, point: SpatialPoint) -> IndexStats:
        coords = point.coords
        # rtree expects (x_min, y_min, ..., x_max, y_max, ...)
        bbox = tuple(coords) + tuple(coords)

        t0 = time.perf_counter_ns()
        self._idx.insert(point.point_id, bbox)
        elapsed = time.perf_counter_ns() - t0

        self._points[point.point_id] = point
        self._mutation_counter += 1

        return IndexStats(
            latency_ns=float(elapsed),
            node_accesses=1,  # approximate
            structural_mutations=1,  # approximate; real count needs instrumentation
        )

    def delete(self, point: SpatialPoint) -> IndexStats:
        coords = point.coords
        bbox = tuple(coords) + tuple(coords)

        t0 = time.perf_counter_ns()
        self._idx.delete(point.point_id, bbox)
        elapsed = time.perf_counter_ns() - t0

        self._points.pop(point.point_id, None)

        return IndexStats(latency_ns=float(elapsed), structural_mutations=1)

    def range_query(self, region: SpatialRegion) -> tuple[list[SpatialPoint], IndexStats]:
        bbox = tuple(region.lo) + tuple(region.hi)

        t0 = time.perf_counter_ns()
        ids = list(self._idx.intersection(bbox))
        elapsed = time.perf_counter_ns() - t0

        results = [self._points[pid] for pid in ids if pid in self._points]

        return results, IndexStats(
            latency_ns=float(elapsed),
            node_accesses=max(1, len(ids)),
        )

    def point_query(self, point: SpatialPoint) -> tuple[SpatialPoint | None, IndexStats]:
        coords = point.coords
        bbox = tuple(coords) + tuple(coords)

        t0 = time.perf_counter_ns()
        ids = list(self._idx.intersection(bbox))
        elapsed = time.perf_counter_ns() - t0

        found = None
        for pid in ids:
            if pid in self._points and pid == point.point_id:
                found = self._points[pid]
                break

        return found, IndexStats(latency_ns=float(elapsed), node_accesses=1)
