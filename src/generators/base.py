"""
Abstract base class for all workload generators.

Every generator must:
1. Accept a reproducible seed
2. Produce a WorkloadTrace with region annotations
3. Expose its parameters for metadata recording
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from src.models.spatial import (
    SpatialRegion,
    WorkloadTrace,
    build_grid_partition,
)


class BaseGenerator(abc.ABC):
    """Abstract workload generator.

    Subclasses implement _generate() which populates self._trace.
    """

    def __init__(
        self,
        *,
        n_points: int,
        n_queries: int,
        dim: int = 2,
        space_lo: float = 0.0,
        space_hi: float = 1.0,
        grid_dims: tuple[int, ...] | None = None,
        seed: int = 42,
    ):
        self.n_points = n_points
        self.n_queries = n_queries
        self.dim = dim
        self.space_lo_val = space_lo
        self.space_hi_val = space_hi
        self.grid_dims = grid_dims or tuple([4] * dim)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build space bounds
        self._lo = np.full(dim, space_lo)
        self._hi = np.full(dim, space_hi)

        # Build partition
        self._regions = build_grid_partition(self._lo, self._hi, self.grid_dims)

        # Output trace
        self._trace = WorkloadTrace(regions=list(self._regions))

    @abc.abstractmethod
    def _generate(self) -> None:
        """Populate self._trace with operations."""
        ...

    def generate(self) -> WorkloadTrace:
        """Run generation and return the annotated trace."""
        self._trace = WorkloadTrace(
            regions=list(self._regions),
            metadata=self.params(),
        )
        self._generate()
        return self._trace

    def params(self) -> dict[str, Any]:
        """Return all generator parameters for reproducibility."""
        return {
            "generator": self.__class__.__name__,
            "n_points": self.n_points,
            "n_queries": self.n_queries,
            "dim": self.dim,
            "space_lo": self.space_lo_val,
            "space_hi": self.space_hi_val,
            "grid_dims": self.grid_dims,
            "seed": self.seed,
        }

    def _find_region(self, coords: np.ndarray) -> int:
        """Return region_id containing coords, or -1."""
        for r in self._regions:
            if (np.all(coords >= r.lo) and np.all(coords <= r.hi)):
                return r.region_id
        return -1
