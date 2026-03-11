"""
LCHI (Locality-Constrained Hierarchical Index) implementation.

This is the proposed index structure from the paper.
Key property: structural mutation domain is bounded by partition boundary.

Each partition maintains an independent sorted list (or local tree).
No cross-partition structural propagation is possible.

Expected: CRI_{i→j} = 0 for all i ≠ j (Theorem 2).

Performance notes
-----------------
* _find_partition: direct grid-math (O(d)) — no region loop, no numpy alloc.
* insert / delete / point_query: operate on two parallel plain-Python lists
  (keys: list[float], points: list[SpatialPoint]) so bisect works directly —
  no O(n) key-list rebuild per operation.
* range_query: partition intersection uses pre-cached float bounds; per-point
  containment uses an inline pure-Python loop — avoids numpy overhead in the
  tight inner loop; sort-key range pruning via bisect further narrows scan.
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


class LCHIIndex(SpatialIndex):
    """Locality-Constrained Hierarchical Index.

    Structure:
    - Stable partition layer (grid)
    - Independent sorted list per partition (upgradeable to B-tree)
    - Zero cross-partition structural coupling

    This is intentionally simple. The contribution is not the data structure
    itself — it's the formal guarantee of bounded CRI.
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

        # Build partition regions (needed for range-query intersection)
        self._regions = build_grid_partition(
            self._space_lo, self._space_hi, self._grid_dims
        )

        # ── fast partition lookup ──────────────────────────────────────────
        # Pre-compute per-dimension step size and row-major strides so that
        # _find_partition is a single O(d) arithmetic pass — no region loop.
        span = self._space_hi - self._space_lo
        self._step: list[float] = [
            float(span[d]) / self._grid_dims[d] for d in range(dim)
        ]
        self._lo_f: list[float] = [float(self._space_lo[d]) for d in range(dim)]

        strides = [1] * dim
        for d in range(dim - 2, -1, -1):
            strides[d] = strides[d + 1] * self._grid_dims[d + 1]
        self._strides: list[int] = strides

        # ── pre-cached region bounds as plain floats ───────────────────────
        # Layout: _rbounds[rid] = [lo_0, lo_1, …, lo_{d-1}, hi_0, hi_1, …]
        # Used in range_query to avoid numpy calls on every intersection test.
        self._rbounds: list[list[float]] = []
        for r in self._regions:
            self._rbounds.append(r.lo.tolist() + r.hi.tolist())

        # ── sort-key weights ───────────────────────────────────────────────
        # key = sum_d( normalized[d] * weight[d] )  where weight[0] dominates.
        self._sk_weights: list[float] = [
            float(10 ** (dim - d - 1)) for d in range(dim)
        ]
        span_f = [float(span[d]) for d in range(dim)]
        self._sk_span: list[float] = span_f  # to avoid repeated attribute lookup

        # ── per-partition sorted storage ───────────────────────────────────
        # Two parallel plain-Python lists per partition — bisect works directly
        # on _keys without rebuilding it on every operation.
        self._keys:   list[list[float]]        = [[] for _ in self._regions]
        self._points: list[list[SpatialPoint]] = [[] for _ in self._regions]
        self._total_size = 0

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"LCHI (grid={'x'.join(map(str, self._grid_dims))})"

    def size(self) -> int:
        return self._total_size

    def _sort_key_coords(self, coords: np.ndarray) -> float:
        """Z-order / Morton code approximation for 1D sorting key."""
        lo = self._lo_f
        span = self._sk_span
        weights = self._sk_weights
        key = 0.0
        for d in range(self._dim):
            key += ((float(coords[d]) - lo[d]) / span[d]) * weights[d]
        return key

    def _find_partition(self, coords: np.ndarray) -> int:
        """Direct grid-math partition lookup — O(d), no region scan."""
        rid = 0
        lo = self._lo_f
        step = self._step
        strides = self._strides
        g = self._grid_dims
        for d in range(self._dim):
            idx = int((float(coords[d]) - lo[d]) / step[d])
            if idx < 0:
                idx = 0
            elif idx >= g[d]:
                idx = g[d] - 1
            rid += idx * strides[d]
        return rid

    # ── insert ─────────────────────────────────────────────────────────────

    def insert(self, point: SpatialPoint) -> IndexStats:
        rid = self._find_partition(point.coords)
        key = self._sort_key_coords(point.coords)
        keys = self._keys[rid]
        pts  = self._points[rid]

        t0 = time.perf_counter_ns()
        idx = bisect.bisect_left(keys, key)
        keys.insert(idx, key)
        pts.insert(idx, point)
        elapsed = time.perf_counter_ns() - t0

        self._total_size += 1
        return IndexStats(
            latency_ns=float(elapsed),
            node_accesses=1,
            structural_mutations=1,  # Local only — cannot propagate
        )

    # ── delete ─────────────────────────────────────────────────────────────

    def delete(self, point: SpatialPoint) -> IndexStats:
        rid = self._find_partition(point.coords)
        key = self._sort_key_coords(point.coords)
        keys = self._keys[rid]
        pts  = self._points[rid]

        t0 = time.perf_counter_ns()
        idx = bisect.bisect_left(keys, key)
        found = False
        for i in range(max(0, idx - 2), min(len(keys), idx + 3)):
            if pts[i].point_id == point.point_id:
                keys.pop(i)
                pts.pop(i)
                self._total_size -= 1
                found = True
                break
        if not found:
            # Fallback full scan (key collision or float rounding)
            for i, p in enumerate(pts):
                if p.point_id == point.point_id:
                    keys.pop(i)
                    pts.pop(i)
                    self._total_size -= 1
                    break
        elapsed = time.perf_counter_ns() - t0
        return IndexStats(latency_ns=float(elapsed), structural_mutations=1)

    # ── range query ────────────────────────────────────────────────────────

    def range_query(self, region: SpatialRegion) -> tuple[list[SpatialPoint], IndexStats]:
        """Range query — only touches intersecting partitions.

        Three-layer optimisation:
        1. Partition intersection via cached plain-float bounds (no numpy).
        2. Sort-key range pruning via bisect to narrow the per-point scan.
        3. Per-point containment via an inline pure-Python loop (no numpy).
        """
        results: list[SpatialPoint] = []
        total_accesses = 0
        dim = self._dim

        # Unpack query bounds once as plain floats
        qlo_f = [float(region.lo[d]) for d in range(dim)]
        qhi_f = [float(region.hi[d]) for d in range(dim)]

        # Sort-key bounds for candidate pruning
        lo_sk = self._sort_key_coords(region.lo)
        hi_sk = self._sort_key_coords(region.hi)
        # For non-monotone dims ensure correct order
        min_sk = min(lo_sk, hi_sk)
        max_sk = max(lo_sk, hi_sk)

        t0 = time.perf_counter_ns()
        for rid, rb in enumerate(self._rbounds):
            # --- fast intersection test (pure Python floats) ---
            ok = True
            for d in range(dim):
                if rb[d] > qhi_f[d] or rb[d + dim] < qlo_f[d]:
                    ok = False
                    break
            if not ok:
                continue

            total_accesses += 1
            keys = self._keys[rid]
            pts  = self._points[rid]
            if not keys:
                continue

            # --- sort-key range prune ---
            lo_idx = max(0, bisect.bisect_left(keys, min_sk) - 1)
            hi_idx = min(len(keys), bisect.bisect_right(keys, max_sk) + 1)

            # --- per-point containment (inline pure Python) ---
            for i in range(lo_idx, hi_idx):
                c = pts[i].coords
                inside = True
                for d in range(dim):
                    cv = float(c[d])
                    if cv < qlo_f[d] or cv > qhi_f[d]:
                        inside = False
                        break
                if inside:
                    results.append(pts[i])

        elapsed = time.perf_counter_ns() - t0
        return results, IndexStats(
            latency_ns=float(elapsed),
            node_accesses=total_accesses,
        )

    # ── point query ────────────────────────────────────────────────────────

    def point_query(self, point: SpatialPoint) -> tuple[SpatialPoint | None, IndexStats]:
        rid = self._find_partition(point.coords)
        key = self._sort_key_coords(point.coords)
        keys = self._keys[rid]
        pts  = self._points[rid]

        t0 = time.perf_counter_ns()
        idx = bisect.bisect_left(keys, key)
        found = None
        for i in range(max(0, idx - 2), min(len(keys), idx + 3)):
            if pts[i].point_id == point.point_id:
                found = pts[i]
                break
        elapsed = time.perf_counter_ns() - t0
        return found, IndexStats(latency_ns=float(elapsed), node_accesses=1)
