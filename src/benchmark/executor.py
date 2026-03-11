"""
Benchmark executor: replays a WorkloadTrace against a SpatialIndex
and feeds measurements into the CRI framework.

This is the central experimental pipeline:
    Generator → Trace → Executor(Index) → CRI Measurement → Results

The executor:
1. Replays every operation in the trace against the index
2. Records latency plus structural stats per operation
3. Feeds all observations to CRIMeasurer
4. Returns full CRI analysis plus per-operation metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from src.indexes.base import SpatialIndex
from src.metrics.cri import CRIMeasurer, CRIResult
from src.models.spatial import OpType, WorkloadTrace


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one (index, workload) pair."""
    index_name: str
    workload_name: str
    workload_params: dict[str, Any]

    # CRI analysis
    cri_result: CRIResult

    # Aggregate latency stats
    insert_latencies_ns: list[float] = field(default_factory=lambda: list[float]())
    query_latencies_ns: list[float] = field(default_factory=lambda: list[float]())

    # Per-region query latencies for detailed analysis
    region_query_latencies: dict[int, list[float]] = field(default_factory=lambda: dict[int, list[float]]())

    # Structural stats
    total_mutations: int = 0
    total_node_accesses: int = 0

    @property
    def mean_insert_latency(self) -> float:
        if not self.insert_latencies_ns:
            return 0.0
        return float(np.mean(self.insert_latencies_ns))

    @property
    def mean_query_latency(self) -> float:
        if not self.query_latencies_ns:
            return 0.0
        return float(np.mean(self.query_latencies_ns))

    @property
    def p99_query_latency(self) -> float:
        if not self.query_latencies_ns:
            return 0.0
        return float(np.percentile(self.query_latencies_ns, 99))

    @property
    def p50_query_latency(self) -> float:
        if not self.query_latencies_ns:
            return 0.0
        return float(np.percentile(self.query_latencies_ns, 50))

    def per_region_mean_latency(self) -> dict[int, float]:
        return {
            rid: float(np.mean(lats)) if lats else 0.0
            for rid, lats in self.region_query_latencies.items()
        }

    def per_region_p99_latency(self) -> dict[int, float]:
        return {
            rid: float(np.percentile(lats, 99)) if len(lats) >= 2 else 0.0
            for rid, lats in self.region_query_latencies.items()
        }

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index_name,
            "workload": self.workload_name,
            "mean_insert_ns": self.mean_insert_latency,
            "mean_query_ns": self.mean_query_latency,
            "p99_query_ns": self.p99_query_latency,
            "p50_query_ns": self.p50_query_latency,
            "total_mutations": self.total_mutations,
            "isolation_score": self.cri_result.isolation_score,
            "isolation_score_raw": self.cri_result.isolation_score_raw,
            "mean_cross_cri": self.cri_result.mean_cross_cri,
            "mean_cross_cri_raw": self.cri_result.mean_cross_cri_raw,
            "epsilon_isolated_0.1": self.cri_result.is_epsilon_isolated(0.1),
            "epsilon_isolated_0.01": self.cri_result.is_epsilon_isolated(0.01),
            "n_significant_pairs": len(self.cri_result.significant_pairs(0.05)),
        }


class BenchmarkExecutor:
    """Replay workload traces against spatial indexes and measure CRI."""

    def __init__(
        self,
        cri_window_size: int = 1000,
        show_progress: bool = True,
    ):
        self.cri_window_size = cri_window_size
        self.show_progress = show_progress

    def run(
        self,
        index: SpatialIndex,
        trace: WorkloadTrace,
        workload_name: str = "unknown",
    ) -> BenchmarkResult:
        """Execute a full benchmark run.

        Args:
            index: the spatial index to test
            trace: the workload trace to replay
            workload_name: label for reporting

        Returns:
            BenchmarkResult with CRI analysis and latency stats.
        """
        n_regions = trace.n_regions
        measurer = CRIMeasurer(
            n_regions=n_regions,
            window_size=self.cri_window_size,
        )

        insert_latencies: list[float] = []
        query_latencies: list[float] = []
        region_query_lats: dict[int, list[float]] = {
            r.region_id: [] for r in trace.regions
        }
        total_mutations = 0
        total_accesses = 0

        ops = trace.operations
        iterator = tqdm(ops, desc=f"{index.name}", disable=not self.show_progress)

        for op in iterator:
            if op.op_type == OpType.INSERT and op.point is not None:
                stats = index.insert(op.point)
                insert_latencies.append(stats.latency_ns)
                total_mutations += stats.structural_mutations
                total_accesses += stats.node_accesses

                measurer.record_update(
                    region_id=op.region_id,
                    timestamp=op.timestamp,
                    structural_mutations=stats.structural_mutations,
                )

            elif op.op_type == OpType.DELETE and op.point is not None:
                stats = index.delete(op.point)
                total_mutations += stats.structural_mutations

                measurer.record_update(
                    region_id=op.region_id,
                    timestamp=op.timestamp,
                    structural_mutations=stats.structural_mutations,
                )

            elif op.op_type == OpType.RANGE_QUERY and op.query_region is not None:
                _results, stats = index.range_query(op.query_region)
                query_latencies.append(stats.latency_ns)
                total_accesses += stats.node_accesses

                rid = op.region_id
                if rid in region_query_lats:
                    region_query_lats[rid].append(stats.latency_ns)

                measurer.record_query(
                    region_id=rid,
                    timestamp=op.timestamp,
                    latency_ns=stats.latency_ns,
                    node_accesses=stats.node_accesses,
                )

            elif op.op_type == OpType.POINT_QUERY and op.point is not None:
                _found, stats = index.point_query(op.point)
                query_latencies.append(stats.latency_ns)
                total_accesses += stats.node_accesses

                rid = op.region_id
                if rid in region_query_lats:
                    region_query_lats[rid].append(stats.latency_ns)

                measurer.record_query(
                    region_id=rid,
                    timestamp=op.timestamp,
                    latency_ns=stats.latency_ns,
                )

        # Compute CRI
        cri_result = measurer.compute_cri()

        return BenchmarkResult(
            index_name=index.name,
            workload_name=workload_name,
            workload_params=trace.metadata,
            cri_result=cri_result,
            insert_latencies_ns=insert_latencies,
            query_latencies_ns=query_latencies,
            region_query_latencies=region_query_lats,
            total_mutations=total_mutations,
            total_node_accesses=total_accesses,
        )
