"""
LSM-based spatial index.

Log-Structured Merge tree adapted for spatial data.
Writes go to an in-memory buffer (memtable); when full, the buffer
is frozen and merged into sorted on-disk levels.

For spatial data, each level is a sorted run keyed by space-filling
curve (Z-order). Range queries must check all levels.

This models modern spatial storage engines (e.g., based on RocksDB).

Expected CRI behavior:
- Writes are buffered → low immediate interference
- Compaction merges levels → periodic global restructuring
- CRI should be INTERMITTENT (spiky) rather than continuous

Baseline B4 in the paper.
"""

from __future__ import annotations

import bisect
import time

import numpy as np

from src.indexes.base import IndexStats, SpatialIndex
from src.models.spatial import SpatialPoint, SpatialRegion


class LSMSpatialIndex(SpatialIndex):
    """LSM-tree based spatial index with Z-order key.

    Structure:
    - Level 0: mutable memtable (sorted list)
    - Level 1..L: immutable sorted runs
    - Compaction: when level k has T runs, merge into level k+1

    Spatial queries: scan all levels, merge results.
    """

    def __init__(
        self,
        dim: int = 2,
        memtable_capacity: int = 1000,
        level_ratio: int = 4,
        max_levels: int = 5,
    ):
        self._dim = dim
        self._memtable_capacity = memtable_capacity
        self._level_ratio = level_ratio
        self._max_levels = max_levels

        # Level 0: mutable sorted list (memtable)
        self._memtable: list[tuple[int, SpatialPoint]] = []  # (z_key, point)

        # Levels 1..L: list of immutable sorted runs
        self._levels: list[list[list[tuple[int, SpatialPoint]]]] = [
            [] for _ in range(max_levels)
        ]

        self._total_size = 0
        self._compaction_mutations = 0

    @property
    def name(self) -> str:
        return f"LSM-Spatial (mem={self._memtable_capacity})"

    def size(self) -> int:
        return self._total_size

    def _z_order_key(self, coords: np.ndarray) -> int:
        """Compute Z-order (Morton code) key for spatial sorting.

        Interleaves bits of quantized coordinates for a 2D space.
        """
        # Quantize to 16-bit integers
        bits = 16
        max_val = (1 << bits) - 1
        quantized = []
        for d in range(self._dim):
            v = int(np.clip(coords[d], 0, 1) * max_val)
            quantized.append(v)

        # Interleave bits
        result = 0
        for bit in range(bits):
            for d in range(self._dim):
                result |= ((quantized[d] >> bit) & 1) << (bit * self._dim + d)
        return result

    def _flush_memtable(self) -> int:
        """Flush memtable to Level 0, trigger compaction if needed."""
        if not self._memtable:
            return 0

        # Sort the memtable (should already be sorted, but ensure)
        frozen_run = sorted(self._memtable, key=lambda x: x[0])
        self._memtable = []

        # Add to Level 0
        self._levels[0].append(frozen_run)
        mutations = len(frozen_run)

        # Check for compaction cascade
        for lvl in range(self._max_levels - 1):
            if len(self._levels[lvl]) >= self._level_ratio:
                mutations += self._compact_level(lvl)

        return mutations

    def _compact_level(self, level: int) -> int:
        """Merge all runs in level into a single run in level+1."""
        if not self._levels[level]:
            return 0

        # Merge all runs at this level
        all_entries: list[tuple[int, SpatialPoint]] = []
        for run in self._levels[level]:
            all_entries.extend(run)

        all_entries.sort(key=lambda x: x[0])
        self._levels[level] = []

        # Push to next level
        next_level = level + 1
        if next_level < self._max_levels:
            self._levels[next_level].append(all_entries)

        mutations = len(all_entries)
        self._compaction_mutations += mutations
        return mutations

    def insert(self, point: SpatialPoint) -> IndexStats:
        z_key = self._z_order_key(point.coords)

        t0 = time.perf_counter_ns()

        # Insert into memtable (sorted)
        keys = [k for k, _ in self._memtable]
        idx = bisect.bisect_left(keys, z_key)
        self._memtable.insert(idx, (z_key, point))

        mutations = 0
        if len(self._memtable) >= self._memtable_capacity:
            mutations = self._flush_memtable()

        elapsed = time.perf_counter_ns() - t0
        self._total_size += 1

        return IndexStats(
            latency_ns=float(elapsed),
            node_accesses=1,
            structural_mutations=mutations,
        )

    def delete(self, point: SpatialPoint) -> IndexStats:
        t0 = time.perf_counter_ns()

        # Tombstone approach: search and remove from whichever level
        # For simplicity, scan all levels
        found = False

        # Check memtable
        for i, (_k, p) in enumerate(self._memtable):
            if p.point_id == point.point_id:
                self._memtable.pop(i)
                found = True
                break

        if not found:
            for lvl in self._levels:
                for run in lvl:
                    for i, (_k, p) in enumerate(run):
                        if p.point_id == point.point_id:
                            run.pop(i)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if found:
            self._total_size -= 1

        elapsed = time.perf_counter_ns() - t0
        return IndexStats(latency_ns=float(elapsed), structural_mutations=1)

    @staticmethod
    def _run_keys(run: list[tuple[int, SpatialPoint]]) -> list[int]:
        """Extract Z-order keys from a sorted run (used for bisect narrowing)."""
        return [entry[0] for entry in run]

    def range_query(self, region: SpatialRegion) -> tuple[list[SpatialPoint], IndexStats]:
        """Range query: must scan ALL levels — this is the LSM cost."""
        results: list[SpatialPoint] = []
        accesses = 0

        t0 = time.perf_counter_ns()

        # Z-order key bounds for the query rectangle.
        # For any point (x,y) in [region.lo, region.hi], the Z-order key satisfies
        # z(region.lo) ≤ z(x,y) ≤ z(region.hi) because the "spread-out" function
        # used by _z_order_key is monotone in each coordinate independently.
        z_lo = self._z_order_key(region.lo)
        z_hi = self._z_order_key(region.hi)

        # Scan memtable (bisect-narrowed; memtable is kept sorted by insert())
        mem_keys = self._run_keys(self._memtable)
        lo_m = bisect.bisect_left(mem_keys, z_lo)
        hi_m = bisect.bisect_right(mem_keys, z_hi)
        accesses += hi_m - lo_m
        for _, pt in self._memtable[lo_m:hi_m]:
            if region.contains_point(pt):
                results.append(pt)

        # Scan all levels (bisect-narrowed per run; runs are immutable and sorted)
        for lvl in self._levels:
            for run in lvl:
                run_keys = self._run_keys(run)
                lo_r = bisect.bisect_left(run_keys, z_lo)
                hi_r = bisect.bisect_right(run_keys, z_hi)
                accesses += hi_r - lo_r
                for _, pt in run[lo_r:hi_r]:
                    if region.contains_point(pt):
                        results.append(pt)

        elapsed = time.perf_counter_ns() - t0

        return results, IndexStats(
            latency_ns=float(elapsed),
            node_accesses=accesses,
        )

    def point_query(self, point: SpatialPoint) -> tuple[SpatialPoint | None, IndexStats]:
        z_key = self._z_order_key(point.coords)

        t0 = time.perf_counter_ns()

        # Check memtable first (most recent)
        keys = [k for k, _ in self._memtable]
        idx = bisect.bisect_left(keys, z_key)
        for i in range(max(0, idx - 1), min(len(self._memtable), idx + 2)):
            if self._memtable[i][1].point_id == point.point_id:
                elapsed = time.perf_counter_ns() - t0
                return self._memtable[i][1], IndexStats(latency_ns=float(elapsed))

        # Check levels (newest first)
        for lvl in self._levels:
            for run in reversed(lvl):
                r_keys = [k for k, _ in run]
                ri = bisect.bisect_left(r_keys, z_key)
                for i in range(max(0, ri - 1), min(len(run), ri + 2)):
                    if run[i][1].point_id == point.point_id:
                        elapsed = time.perf_counter_ns() - t0
                        return run[i][1], IndexStats(latency_ns=float(elapsed))

        elapsed = time.perf_counter_ns() - t0
        return None, IndexStats(latency_ns=float(elapsed))
