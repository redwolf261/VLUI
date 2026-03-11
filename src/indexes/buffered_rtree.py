"""
Buffered R-tree index.

Modification of the global R-tree: inserts are buffered and applied
in batches. This tests whether buffering reduces CRI (hypothesis: no,
because structural propagation still occurs, just deferred).

This is baseline B2 in the paper.
"""

from __future__ import annotations

import time
from collections import deque

from src.indexes.base import IndexStats, SpatialIndex
from src.models.spatial import SpatialPoint, SpatialRegion


class BufferedRTreeIndex(SpatialIndex):
    """R-tree with write buffer — deferred bulk insertion.

    Buffer accumulates inserts; when buffer reaches capacity,
    all buffered points are flushed into the R-tree at once.

    Expected: CRI slightly reduced vs naive R-tree (amortized splits),
    but still > 0 because the global tree structure is shared.
    """

    def __init__(
        self,
        dim: int = 2,
        leaf_capacity: int = 50,
        buffer_capacity: int = 1000,
    ):
        from rtree import index as rtree_index  # type: ignore[import-untyped]

        p = rtree_index.Property()
        p.dimension = dim
        p.leaf_capacity = leaf_capacity
        p.fill_factor = 0.7

        self._dim = dim
        self._idx = rtree_index.Index(properties=p)
        self._points: dict[int, SpatialPoint] = {}
        self._buffer: deque[SpatialPoint] = deque()
        self._buffer_capacity = buffer_capacity
        self._pending_mutations = 0

    @property
    def name(self) -> str:
        return f"Buffered R-tree (buf={self._buffer_capacity})"

    def size(self) -> int:
        return len(self._points) + len(self._buffer)

    def _flush_buffer(self) -> int:
        """Flush all buffered points into the R-tree.

        Returns number of structural mutations (approximated).
        """
        mutations = 0
        while self._buffer:
            pt = self._buffer.popleft()
            bbox = tuple(pt.coords) + tuple(pt.coords)
            self._idx.insert(pt.point_id, bbox)
            self._points[pt.point_id] = pt
            mutations += 1
        return mutations

    def insert(self, point: SpatialPoint) -> IndexStats:
        t0 = time.perf_counter_ns()

        self._buffer.append(point)
        mutations = 0

        if len(self._buffer) >= self._buffer_capacity:
            mutations = self._flush_buffer()

        elapsed = time.perf_counter_ns() - t0

        return IndexStats(
            latency_ns=float(elapsed),
            node_accesses=1,
            structural_mutations=mutations,
        )

    def delete(self, point: SpatialPoint) -> IndexStats:
        # Check buffer first
        t0 = time.perf_counter_ns()
        for i, bp in enumerate(self._buffer):
            if bp.point_id == point.point_id:
                del self._buffer[i]
                elapsed = time.perf_counter_ns() - t0
                return IndexStats(latency_ns=float(elapsed))

        # Delete from tree
        if point.point_id in self._points:
            bbox = tuple(point.coords) + tuple(point.coords)
            self._idx.delete(point.point_id, bbox)
            del self._points[point.point_id]

        elapsed = time.perf_counter_ns() - t0
        return IndexStats(latency_ns=float(elapsed), structural_mutations=1)

    def range_query(self, region: SpatialRegion) -> tuple[list[SpatialPoint], IndexStats]:
        # Must flush buffer first to get correct results
        t0 = time.perf_counter_ns()

        if self._buffer:
            self._flush_buffer()

        bbox = tuple(region.lo) + tuple(region.hi)
        ids = list(self._idx.intersection(bbox))
        elapsed = time.perf_counter_ns() - t0

        results = [self._points[pid] for pid in ids if pid in self._points]

        return results, IndexStats(
            latency_ns=float(elapsed),
            node_accesses=max(1, len(ids)),
        )

    def point_query(self, point: SpatialPoint) -> tuple[SpatialPoint | None, IndexStats]:
        t0 = time.perf_counter_ns()

        # Check buffer
        for bp in self._buffer:
            if bp.point_id == point.point_id:
                elapsed = time.perf_counter_ns() - t0
                return bp, IndexStats(latency_ns=float(elapsed), node_accesses=1)

        # Check tree
        if self._buffer:
            self._flush_buffer()

        coords = point.coords
        bbox = tuple(coords) + tuple(coords)
        ids = list(self._idx.intersection(bbox))
        elapsed = time.perf_counter_ns() - t0

        found = None
        for pid in ids:
            if pid == point.point_id and pid in self._points:
                found = self._points[pid]
                break

        return found, IndexStats(latency_ns=float(elapsed), node_accesses=1)
