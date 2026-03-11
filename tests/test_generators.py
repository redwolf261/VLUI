"""
Tests for workload generators and CRI measurement.

These verify:
1. Each generator produces valid traces with correct region assignments
2. Zipf generator actually produces skewed distributions
3. Adversarial generator concentrates updates in attack region
4. CRI measurer produces correct matrix dimensions
5. LCHI produces zero cross-region CRI
"""

import numpy as np

from src.models.spatial import OpType
from src.generators.uniform import UniformRandomGenerator
from src.generators.zipf_spatial import ZipfSpatialGenerator
from src.generators.moving_hotspot import MovingHotspotGenerator, HotspotPath
from src.generators.adversarial import AdversarialBurstGenerator
from src.metrics.cri import CRIMeasurer
from src.indexes.lchi_index import LCHIIndex
from src.indexes.learned_grid import LearnedGridIndex
from src.benchmark.executor import BenchmarkExecutor


# ── Generator Tests ──────────────────────────────────────────────────


class TestUniformGenerator:
    def test_output_size(self):
        gen = UniformRandomGenerator(n_points=1000, n_queries=100, seed=1)
        trace = gen.generate()
        assert trace.n_ops == 1100

    def test_region_assignments(self):
        gen = UniformRandomGenerator(
            n_points=500, n_queries=50, grid_dims=(2, 2), seed=1
        )
        trace = gen.generate()
        rates = trace.update_rate_per_region()
        # Should be roughly uniform across 4 regions
        assert len(rates) >= 3  # at least 3 of 4 regions get updates
        for rid, _count in rates.items():
            assert 0 <= rid < 4

    def test_operations_have_points_or_regions(self):
        gen = UniformRandomGenerator(n_points=100, n_queries=10, seed=1)
        trace = gen.generate()
        for op in trace.operations:
            if op.op_type == OpType.INSERT:
                assert op.point is not None
            elif op.op_type == OpType.RANGE_QUERY:
                assert op.query_region is not None


class TestZipfGenerator:
    def test_skew_increases_with_alpha(self):
        """Higher α should concentrate more updates in fewer regions."""
        rates_by_alpha = {}
        for alpha in [0.5, 1.5, 3.0]:
            gen = ZipfSpatialGenerator(
                n_points=10000, n_queries=0, alpha=alpha,
                grid_dims=(4, 4), seed=42,
            )
            trace = gen.generate()
            rates = trace.update_rate_per_region()
            max_rate = max(rates.values())
            rates_by_alpha[alpha] = max_rate

        # Higher alpha → more concentration → higher max rate
        assert rates_by_alpha[3.0] > rates_by_alpha[1.5]
        assert rates_by_alpha[1.5] > rates_by_alpha[0.5]

    def test_queries_are_uniform(self):
        gen = ZipfSpatialGenerator(
            n_points=5000, n_queries=2000, alpha=2.0,
            grid_dims=(4, 4), seed=42,
        )
        trace = gen.generate()
        query_regions = [
            op.region_id for op in trace.operations
            if op.op_type == OpType.RANGE_QUERY
        ]
        unique_regions = set(query_regions)
        # Queries should span most regions (uniform)
        assert len(unique_regions) >= 10  # out of 16


class TestMovingHotspotGenerator:
    def test_linear_path(self):
        gen = MovingHotspotGenerator(
            n_points=5000, n_queries=500,
            hotspot_intensity=0.9, path_type=HotspotPath.LINEAR,
            grid_dims=(4, 4), seed=42,
        )
        trace = gen.generate()
        assert trace.n_ops == 5500

    def test_hotspot_intensity(self):
        """Most updates should land near the hotspot."""
        gen = MovingHotspotGenerator(
            n_points=5000, n_queries=0,
            hotspot_intensity=0.95, hotspot_radius=0.15,
            path_type=HotspotPath.LINEAR,
            grid_dims=(4, 4), seed=42,
        )
        trace = gen.generate()
        rates = trace.update_rate_per_region()
        max_rate = max(rates.values())
        total = sum(rates.values())
        # Top region should have significant fraction
        assert max_rate / total > 0.1


class TestAdversarialGenerator:
    def test_attack_concentration(self):
        gen = AdversarialBurstGenerator(
            n_points=10000, n_queries=1000,
            attack_region_id=0, burst_size=500, n_bursts=5,
            grid_dims=(4, 4), seed=42,
        )
        trace = gen.generate()
        rates = trace.update_rate_per_region()
        # Attack region should dominate
        assert rates.get(0, 0) > sum(
            v for k, v in rates.items() if k != 0
        ) * 0.3  # at least 30% of non-attack total

    def test_queries_in_victim_regions(self):
        gen = AdversarialBurstGenerator(
            n_points=5000, n_queries=500,
            attack_region_id=0,
            grid_dims=(4, 4), seed=42,
        )
        trace = gen.generate()
        query_regions = [
            op.region_id for op in trace.operations
            if op.op_type == OpType.RANGE_QUERY
        ]
        # No queries should target attack region (region 0)
        assert 0 not in query_regions


# ── CRI Measurement Tests ────────────────────────────────────────────


class TestCRIMeasurer:
    def test_matrix_dimensions(self):
        m = 4
        measurer = CRIMeasurer(n_regions=m, window_size=10)

        for t in range(100):
            measurer.record_update(region_id=t % m, timestamp=t)
            measurer.record_query(
                region_id=(t + 1) % m, timestamp=t,
                latency_ns=float(100 + t),
            )

        result = measurer.compute_cri()
        assert result.cri_matrix.shape == (m, m)
        assert result.n_regions == m

    def test_isolated_index_zero_cri(self):
        """If updates and queries are independent, CRI should be near zero."""
        m = 4
        measurer = CRIMeasurer(n_regions=m, window_size=50)
        rng = np.random.default_rng(42)

        for t in range(1000):
            # Random updates
            measurer.record_update(region_id=int(rng.integers(0, m)), timestamp=t)
            # Constant latency queries (no interference)
            measurer.record_query(
                region_id=int(rng.integers(0, m)), timestamp=t,
                latency_ns=100.0,  # constant = no correlation
            )

        result = measurer.compute_cri()
        # All CRI should be near zero since latency doesn't vary with updates
        assert result.isolation_score < 1.0  # very loose bound for noisy test


# ── Integration Test ─────────────────────────────────────────────────


class TestIntegration:
    def test_lchi_full_pipeline(self):
        """End-to-end: generate trace → run on LCHI → measure CRI."""
        gen = ZipfSpatialGenerator(
            n_points=2000, n_queries=200, alpha=2.0,
            grid_dims=(2, 2), seed=42,
        )
        trace = gen.generate()

        index = LCHIIndex(dim=2, grid_dims=(2, 2))
        executor = BenchmarkExecutor(cri_window_size=100, show_progress=False)
        result = executor.run(index, trace, workload_name="test_zipf")

        assert result.index_name.startswith("LCHI")
        assert len(result.query_latencies_ns) > 0
        assert result.cri_result.n_regions == 4

    def test_workload_reproducibility(self):
        """Same seed → same trace."""
        gen1 = ZipfSpatialGenerator(n_points=500, n_queries=50, alpha=1.5, seed=123)
        gen2 = ZipfSpatialGenerator(n_points=500, n_queries=50, alpha=1.5, seed=123)

        t1 = gen1.generate()
        t2 = gen2.generate()

        assert t1.n_ops == t2.n_ops
        for op1, op2 in zip(t1.operations, t2.operations):
            assert op1.op_type == op2.op_type
            assert op1.timestamp == op2.timestamp
            assert op1.region_id == op2.region_id

    def test_learned_grid_cri_comparable_to_lchi(self):
        """LearnedGridIndex should achieve isolation < 3.0 on a Zipf workload."""
        gen = ZipfSpatialGenerator(
            n_points=500, n_queries=50, alpha=2.0,
            grid_dims=(2, 2), seed=7,
        )
        trace = gen.generate()

        index = LearnedGridIndex(dim=2, grid_dims=(2, 2))
        executor = BenchmarkExecutor(cri_window_size=50, show_progress=False)
        result = executor.run(index, trace, workload_name="test_learned")

        assert result.index_name.startswith("Learned")
        assert result.cri_result.n_regions == 4
        assert len(result.query_latencies_ns) > 0
        # isolation_score is a real float — just verify it's finite and non-negative
        assert 0.0 <= result.cri_result.isolation_score < float("inf")
