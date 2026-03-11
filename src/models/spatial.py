"""
Core spatial data models for the VLUI research framework.

These models define the fundamental abstractions:
- SpatialPoint: A point in R^d with metadata
- SpatialRegion: An axis-aligned bounding box defining a partition region
- Operation: A timestamped insert/delete/query event
- WorkloadTrace: An ordered sequence of operations with region annotations
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class OpType(enum.Enum):
    """Operation types in a spatial workload."""
    INSERT = "insert"
    DELETE = "delete"
    RANGE_QUERY = "range_query"
    POINT_QUERY = "point_query"


@dataclass(frozen=True, slots=True)
class SpatialPoint:
    """A point in R^d.

    Attributes:
        coords: coordinate vector, shape (d,)
        point_id: unique identifier (monotonic within a trace)
        timestamp: logical time of creation
    """
    coords: NDArray[np.float64]
    point_id: int
    timestamp: int = 0

    @property
    def dim(self) -> int:
        return self.coords.shape[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpatialPoint):
            return NotImplemented
        return self.point_id == other.point_id

    def __hash__(self) -> int:
        return hash(self.point_id)


@dataclass(frozen=True, slots=True)
class SpatialRegion:
    """Axis-aligned bounding box in R^d.

    Attributes:
        lo: lower-left corner, shape (d,)
        hi: upper-right corner, shape (d,)
        region_id: identifier for partition membership
    """
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    region_id: int = 0

    @property
    def dim(self) -> int:
        return self.lo.shape[0]

    @property
    def volume(self) -> float:
        return float(np.prod(self.hi - self.lo))

    @property
    def center(self) -> NDArray[np.float64]:
        return (self.lo + self.hi) / 2.0

    def contains_point(self, p: SpatialPoint) -> bool:
        return bool(np.all(p.coords >= self.lo) and np.all(p.coords <= self.hi))

    def intersects(self, other: SpatialRegion) -> bool:
        return bool(np.all(self.lo <= other.hi) and np.all(self.hi >= other.lo))

    def intersection_volume(self, other: SpatialRegion) -> float:
        lo = np.maximum(self.lo, other.lo)
        hi = np.minimum(self.hi, other.hi)
        if np.any(lo > hi):
            return 0.0
        return float(np.prod(hi - lo))


@dataclass(frozen=True, slots=True)
class Operation:
    """A single timestamped operation in the workload trace.

    Attributes:
        op_type: INSERT, DELETE, RANGE_QUERY, or POINT_QUERY
        timestamp: logical clock tick
        point: the point being inserted/deleted/queried (None for range query)
        query_region: bounding box for range queries (None for point ops)
        region_id: which partition region this operation targets
    """
    op_type: OpType
    timestamp: int
    point: SpatialPoint | None = None
    query_region: SpatialRegion | None = None
    region_id: int = -1  # -1 = unassigned


@dataclass
class WorkloadTrace:
    """An ordered sequence of operations with region annotations.

    This is the output of every generator. It carries enough metadata
    to compute CRI: each operation is tagged with its target region,
    and the trace records the partition definition used.

    Attributes:
        operations: ordered list of operations
        regions: the partition regions R_1, ..., R_m
        metadata: generator parameters for reproducibility
    """
    operations: list[Operation] = field(default_factory=list)
    regions: list[SpatialRegion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_ops(self) -> int:
        return len(self.operations)

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def ops_in_region(self, region_id: int) -> list[Operation]:
        return [op for op in self.operations if op.region_id == region_id]

    def updates_in_region(self, region_id: int) -> list[Operation]:
        return [
            op for op in self.operations
            if op.region_id == region_id
            and op.op_type in (OpType.INSERT, OpType.DELETE)
        ]

    def queries_in_region(self, region_id: int) -> list[Operation]:
        return [
            op for op in self.operations
            if op.region_id == region_id
            and op.op_type in (OpType.RANGE_QUERY, OpType.POINT_QUERY)
        ]

    def update_rate_per_region(self) -> dict[int, int]:
        """Count updates per region. U_i(t) aggregate."""
        rates: dict[int, int] = {}
        for op in self.operations:
            if op.op_type in (OpType.INSERT, OpType.DELETE):
                rates[op.region_id] = rates.get(op.region_id, 0) + 1
        return rates

    def append(self, op: Operation) -> None:
        self.operations.append(op)

    def extend(self, ops: Sequence[Operation]) -> None:
        self.operations.extend(ops)


def build_grid_partition(
    space_lo: NDArray[np.float64],
    space_hi: NDArray[np.float64],
    grid_dims: Sequence[int],
) -> list[SpatialRegion]:
    """Create a uniform grid partition of R^d.

    Args:
        space_lo: lower bound of space, shape (d,)
        space_hi: upper bound of space, shape (d,)
        grid_dims: number of cells along each dimension, e.g. (4, 4) for 2D

    Returns:
        List of SpatialRegion, one per grid cell, with region_id assigned.
    """
    d = len(grid_dims)
    edges = []
    for dim in range(d):
        edges.append(np.linspace(space_lo[dim], space_hi[dim], grid_dims[dim] + 1))

    regions: list[SpatialRegion] = []
    region_id = 0

    # Generate all grid cells via cartesian product of dimension indices
    indices = [range(g) for g in grid_dims]
    import itertools
    for cell_idx in itertools.product(*indices):
        lo = np.array([edges[dim][cell_idx[dim]] for dim in range(d)])
        hi = np.array([edges[dim][cell_idx[dim] + 1] for dim in range(d)])
        regions.append(SpatialRegion(lo=lo, hi=hi, region_id=region_id))
        region_id += 1

    return regions


def assign_region(point: SpatialPoint, regions: list[SpatialRegion]) -> int:
    """Find the region containing a point. Returns region_id or -1."""
    for r in regions:
        if r.contains_point(point):
            return r.region_id
    return -1
