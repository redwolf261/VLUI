"""
Learned Grid Index — partition-confined index with per-partition linear CDF model.

Design: identical partition layout to LCHIIndex, but each partition maintains a
lightweight linear model that predicts approximate insert position from the sort
key. This reduces the binary-search window and acts as an accelerator.

The model is retrained every `refit_every` inserts per partition via
  np.polyfit(keys, positions, 1)  → [slope, intercept]  (key → predicted index)

CRI should be comparably low to LCHI (partition-confined), with possibly lower
latency due to learned lookup, or higher during refit bursts.

Baseline B5 in the paper (learned index comparator).
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

# Window half-width around the model's predicted position before falling back
# to standard bisect. Within ±_PRED_WINDOW the search is O(1) expected.
_PRED_WINDOW = 32


class LearnedGridIndex(SpatialIndex):
    """Grid-partitioned index with per-partition linear CDF learned model.

    Each partition holds an independently sorted list of (key, point) pairs.
    A linear model predicts approximate insertion rank from the sort key,
    replacing cold bisect with a predicted-position + small-window scan.
    Cross-partition structural independence is preserved → CRI_{i→j} ≈ 0.
    """

    def __init__(
        self,
        dim: int = 2,
        space_lo: float = 0.0,
        space_hi: float = 1.0,
        grid_dims: tuple[int, ...] | None = None,
        refit_every: int = 50,
    ):
        self._dim = dim
        self._space_lo = np.full(dim, space_lo)
        self._space_hi = np.full(dim, space_hi)
        self._grid_dims = grid_dims or tuple([4] * dim)
        self._refit_every = refit_every

        self._regions = build_grid_partition(
            self._space_lo, self._space_hi, self._grid_dims
        )
        n_partitions = len(self._regions)

        # ── fast partition lookup (same arithmetic as LCHIIndex) ───────────
        span = self._space_hi - self._space_lo
        self._step: list[float] = [float(span[d]) / self._grid_dims[d] for d in range(dim)]
        self._lo_f: list[float] = [float(self._space_lo[d]) for d in range(dim)]
        strides = [1] * dim
        for d in range(dim - 2, -1, -1):
            strides[d] = strides[d + 1] * self._grid_dims[d + 1]
        self._strides: list[int] = strides

        # ── pre-cached region bounds as plain floats (for range_query) ─────
        self._rbounds: list[list[float]] = [
            r.lo.tolist() + r.hi.tolist() for r in self._regions
        ]

        # ── sort-key weights ───────────────────────────────────────────────
        self._sk_weights: list[float] = [float(10 ** (dim - d - 1)) for d in range(dim)]
        self._sk_span: list[float] = [float(span[d]) for d in range(dim)]

        # ── per-partition sorted storage ───────────────────────────────────
        self._keys:   list[list[float]]        = [[] for _ in range(n_partitions)]
        self._points: list[list[SpatialPoint]] = [[] for _ in range(n_partitions)]

        # ── per-partition linear model: (slope, intercept) or None ─────────
        self._models: list[tuple[float, float] | None] = [None] * n_partitions
        self._insert_counts: list[int] = [0] * n_partitions

        self._total_size = 0

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"LearnedGrid (grid={'x'.join(map(str, self._grid_dims))}, refit={self._refit_every})"

    def size(self) -> int:
        return self._total_size

    def _sort_key_coords(self, coords: np.ndarray) -> float:
        lo = self._lo_f
        span = self._sk_span
        weights = self._sk_weights
        key = 0.0
        for d in range(self._dim):
            key += ((float(coords[d]) - lo[d]) / span[d]) * weights[d]
        return key

    def _find_partition(self, coords: np.ndarray) -> int:
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

    def _refit(self, rid: int) -> None:
        """Fit a linear model: predicted_rank = slope * key + intercept.

        Uses np.polyfit(keys, positions, 1) so the model maps from key value
        to approximate list index. O(n_partition) — called every refit_every
        inserts, so amortized O(1) per insert.
        """
        keys = self._keys[rid]
        n = len(keys)
        if n < 2:
            self._models[rid] = None
            return
        positions = list(range(n))
        coeffs = np.polyfit(keys, positions, 1)
        self._models[rid] = (float(coeffs[0]), float(coeffs[1]))

    def _predict_pos(self, rid: int, key: float) -> int | None:
        """Return model-predicted insert position, or None if no model."""
        model = self._models[rid]
        if model is None:
            return None
        slope, intercept = model
        pred = slope * key + intercept
        return int(pred + 0.5)

    # ── insert ─────────────────────────────────────────────────────────────

    def insert(self, point: SpatialPoint) -> IndexStats:
        rid = self._find_partition(point.coords)
        key = self._sort_key_coords(point.coords)
        keys = self._keys[rid]
        pts = self._points[rid]

        t0 = time.perf_counter_ns()

        # Use model prediction to narrow the bisect window
        pred = self._predict_pos(rid, key)
        if pred is not None:
            lo_w = max(0, min(len(keys), pred - _PRED_WINDOW))
            hi_w = max(0, min(len(keys), pred + _PRED_WINDOW))
            # Verify the prediction window covers the key, else full bisect
            if (lo_w == 0 or keys[lo_w - 1] <= key) and (hi_w == len(keys) or keys[hi_w] >= key):
                idx = lo_w + bisect.bisect_left(keys[lo_w:hi_w], key)
            else:
                idx = bisect.bisect_left(keys, key)
        else:
            idx = bisect.bisect_left(keys, key)

        keys.insert(idx, key)
        pts.insert(idx, point)
        self._total_size += 1

        self._insert_counts[rid] += 1
        mutations = 0
        if self._insert_counts[rid] % self._refit_every == 0:
            self._refit(rid)
            mutations = 1  # model refit = structural mutation

        elapsed = time.perf_counter_ns() - t0
        return IndexStats(
            latency_ns=float(elapsed),
            node_accesses=1,
            structural_mutations=mutations,
        )

    # ── delete ─────────────────────────────────────────────────────────────

    def delete(self, point: SpatialPoint) -> IndexStats:
        rid = self._find_partition(point.coords)
        key = self._sort_key_coords(point.coords)
        keys = self._keys[rid]
        pts = self._points[rid]

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
        """Partition-confined range query — identical logic to LCHIIndex.

        Only intersecting partitions are touched; no cross-partition access.
        Sort-key bounds prune the per-partition candidate window via bisect.
        """
        results: list[SpatialPoint] = []
        total_accesses = 0
        dim = self._dim

        qlo_f = [float(region.lo[d]) for d in range(dim)]
        qhi_f = [float(region.hi[d]) for d in range(dim)]

        lo_sk = self._sort_key_coords(region.lo)
        hi_sk = self._sort_key_coords(region.hi)
        min_sk = min(lo_sk, hi_sk)
        max_sk = max(lo_sk, hi_sk)

        t0 = time.perf_counter_ns()
        for rid, rb in enumerate(self._rbounds):
            ok = True
            for d in range(dim):
                if rb[d] > qhi_f[d] or rb[d + dim] < qlo_f[d]:
                    ok = False
                    break
            if not ok:
                continue

            total_accesses += 1
            keys = self._keys[rid]
            pts = self._points[rid]
            if not keys:
                continue

            lo_idx = max(0, bisect.bisect_left(keys, min_sk) - 1)
            hi_idx = min(len(keys), bisect.bisect_right(keys, max_sk) + 1)

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
        pts = self._points[rid]

        t0 = time.perf_counter_ns()
        idx = bisect.bisect_left(keys, key)
        found = None
        for i in range(max(0, idx - 2), min(len(keys), idx + 3)):
            if pts[i].point_id == point.point_id:
                found = pts[i]
                break
        elapsed = time.perf_counter_ns() - t0
        return found, IndexStats(latency_ns=float(elapsed), node_accesses=1)
