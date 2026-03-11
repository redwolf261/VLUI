"""
Abstract spatial index interface.

Every index under test (R-tree, LCHI, grid, etc.) must implement this
interface so the benchmark harness can measure CRI uniformly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.models.spatial import SpatialPoint, SpatialRegion


@dataclass
class IndexStats:
    """Per-operation statistics returned by the index."""
    latency_ns: float = 0.0
    node_accesses: int = 0
    structural_mutations: int = 0  # splits + merges
    height: int = 0
    total_nodes: int = 0


class SpatialIndex(abc.ABC):
    """Abstract spatial index interface for benchmark evaluation."""

    @abc.abstractmethod
    def insert(self, point: SpatialPoint) -> IndexStats:
        """Insert a point and return operation statistics."""
        ...

    @abc.abstractmethod
    def delete(self, point: SpatialPoint) -> IndexStats:
        """Delete a point and return operation statistics."""
        ...

    @abc.abstractmethod
    def range_query(self, region: SpatialRegion) -> tuple[list[SpatialPoint], IndexStats]:
        """Execute a range query. Returns (results, stats)."""
        ...

    @abc.abstractmethod
    def point_query(self, point: SpatialPoint) -> tuple[SpatialPoint | None, IndexStats]:
        """Find exact point. Returns (found_point_or_none, stats)."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable index name for reporting."""
        ...

    @abc.abstractmethod
    def size(self) -> int:
        """Number of points currently stored."""
        ...

    def bulk_load(self, points: list[SpatialPoint]) -> None:
        """Optional bulk loading. Default: sequential inserts."""
        for p in points:
            self.insert(p)
