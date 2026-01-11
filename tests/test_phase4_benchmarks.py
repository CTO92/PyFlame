"""
Tests for PyFlame Phase 4 Benchmarks Module.

Tests BenchmarkRunner, BenchmarkResult, and related utilities.
"""

import pytest
import sys
import os
import tempfile
import json

# Add Python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# Check if C++ bindings are available
import pyflame
requires_cpp = pytest.mark.skipif(
    not pyflame._CPP_AVAILABLE,
    reason="Requires PyFlame C++ bindings to be built"
)

from pyflame.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
    benchmark,
    timed,
)
from pyflame.benchmarks.results import (
    BenchmarkReport,
    compare_results,
)
from pyflame.benchmarks.models import (
    get_benchmark_model,
    list_benchmark_models,
    get_benchmark_model_info,
    BenchmarkModelInfo,
)


# =============================================================================
# Mock Model
# =============================================================================

class MockModel:
    """Mock model for benchmarking tests."""

    def __init__(self):
        self._eval_mode = False

    def __call__(self, x):
        import numpy as np
        # Simulate some computation
        result = np.sum(x) * 0.1
        return np.random.randn(x.shape[0], 10).astype(np.float32)

    def eval(self):
        self._eval_mode = True

    def train(self):
        self._eval_mode = False


# =============================================================================
# BenchmarkConfig Tests
# =============================================================================

class TestBenchmarkConfig:
    """Test cases for BenchmarkConfig."""

    def test_default_config(self):
        config = BenchmarkConfig()
        assert config.warmup_iterations == 10
        assert config.benchmark_iterations == 100
        assert config.batch_sizes == [1, 8, 32, 64]
        assert config.precision == "fp32"
        assert config.profile_memory is True

    def test_custom_config(self):
        config = BenchmarkConfig(
            warmup_iterations=5,
            benchmark_iterations=50,
            batch_sizes=[2, 4, 8],
        )
        assert config.warmup_iterations == 5
        assert config.benchmark_iterations == 50
        assert config.batch_sizes == [2, 4, 8]


# =============================================================================
# BenchmarkResult Tests
# =============================================================================

class TestBenchmarkResult:
    """Test cases for BenchmarkResult."""

    def test_result_creation(self):
        result = BenchmarkResult(
            name="test_model",
            batch_size=32,
            latency_ms=10.5,
            throughput=3048.0,
            memory_mb=100.0,
            std_dev_ms=1.2,
        )
        assert result.name == "test_model"
        assert result.batch_size == 32
        assert result.latency_ms == 10.5

    def test_result_to_dict(self):
        result = BenchmarkResult(
            name="test",
            batch_size=1,
            latency_ms=5.0,
            throughput=200.0,
            memory_mb=50.0,
            std_dev_ms=0.5,
        )
        d = result.to_dict()

        assert d["name"] == "test"
        assert d["batch_size"] == 1
        assert d["latency_ms"] == 5.0
        assert "throughput" in d


# =============================================================================
# BenchmarkRunner Tests
# =============================================================================

class TestBenchmarkRunner:
    """Test cases for BenchmarkRunner."""

    def test_runner_initialization(self):
        runner = BenchmarkRunner()
        assert runner.config is not None
        assert runner.results == []

    def test_runner_with_config(self):
        config = BenchmarkConfig(benchmark_iterations=10)
        runner = BenchmarkRunner(config)
        assert runner.config.benchmark_iterations == 10

    @requires_cpp
    def test_run_model_benchmark(self):
        config = BenchmarkConfig(
            warmup_iterations=2,
            benchmark_iterations=5,
            batch_sizes=[1, 2],
        )
        runner = BenchmarkRunner(config)
        model = MockModel()

        results = runner.run_model_benchmark(
            "mock_model",
            model,
            input_shape=[10],
        )

        assert len(results) == 2  # Two batch sizes
        assert all(r.name == "mock_model" for r in results)
        assert results[0].batch_size == 1
        assert results[1].batch_size == 2

    @requires_cpp
    def test_run_operation_benchmark(self):
        import numpy as np

        config = BenchmarkConfig(
            warmup_iterations=2,
            benchmark_iterations=5,
        )
        runner = BenchmarkRunner(config)

        def my_operation(x, y):
            return x + y

        result = runner.run_operation_benchmark(
            "add_operation",
            my_operation,
            input_shapes=[[10, 10], [10, 10]],
        )

        assert result.name == "add_operation"
        assert result.latency_ms > 0

    @requires_cpp
    def test_runner_export_json(self):
        config = BenchmarkConfig(
            warmup_iterations=2,
            benchmark_iterations=5,
            batch_sizes=[1],
        )
        runner = BenchmarkRunner(config)
        model = MockModel()

        runner.run_model_benchmark("test", model, [10])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            runner.export_json(path)

            with open(path, 'r') as f:
                data = json.load(f)

            assert "config" in data
            assert "results" in data
            assert len(data["results"]) == 1
        finally:
            os.unlink(path)

    @requires_cpp
    def test_runner_export_csv(self):
        config = BenchmarkConfig(
            warmup_iterations=2,
            benchmark_iterations=5,
            batch_sizes=[1],
        )
        runner = BenchmarkRunner(config)
        model = MockModel()

        runner.run_model_benchmark("test", model, [10])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            runner.export_csv(path)

            with open(path, 'r') as f:
                content = f.read()

            assert "name" in content
            assert "latency_ms" in content
        finally:
            os.unlink(path)

    def test_runner_clear_results(self):
        runner = BenchmarkRunner()
        runner.results.append(BenchmarkResult(
            name="test", batch_size=1, latency_ms=1.0,
            throughput=1000.0, memory_mb=10.0, std_dev_ms=0.1
        ))

        assert len(runner.results) == 1
        runner.clear_results()
        assert len(runner.results) == 0


# =============================================================================
# benchmark() Function Tests
# =============================================================================

class TestBenchmarkFunction:
    """Test cases for benchmark() convenience function."""

    @requires_cpp
    def test_benchmark_function(self):
        model = MockModel()
        results = benchmark(
            model,
            input_shape=[10],
            batch_sizes=[1],
            iterations=5,
            print_results=False,
        )

        assert len(results) == 1
        assert results[0].name == "model"


# =============================================================================
# BenchmarkReport Tests
# =============================================================================

class TestBenchmarkReport:
    """Test cases for BenchmarkReport."""

    def test_report_creation(self):
        results = [
            BenchmarkResult(
                name="model1", batch_size=1, latency_ms=5.0,
                throughput=200.0, memory_mb=50.0, std_dev_ms=0.5
            ),
        ]
        report = BenchmarkReport(title="Test Report", results=results)

        assert report.title == "Test Report"
        assert len(report.results) == 1
        assert report.timestamp != ""

    def test_report_summary(self):
        results = [
            BenchmarkResult(
                name="model1", batch_size=1, latency_ms=5.0,
                throughput=200.0, memory_mb=50.0, std_dev_ms=0.5
            ),
        ]
        report = BenchmarkReport(title="Test Report", results=results)
        summary = report.summary()

        assert "Test Report" in summary
        assert "model1" in summary

    def test_report_to_dict(self):
        results = [
            BenchmarkResult(
                name="model1", batch_size=1, latency_ms=5.0,
                throughput=200.0, memory_mb=50.0, std_dev_ms=0.5
            ),
        ]
        report = BenchmarkReport(title="Test", results=results)
        d = report.to_dict()

        assert d["title"] == "Test"
        assert len(d["results"]) == 1

    def test_report_save_load_json(self):
        results = [
            BenchmarkResult(
                name="model1", batch_size=1, latency_ms=5.0,
                throughput=200.0, memory_mb=50.0, std_dev_ms=0.5
            ),
        ]
        report = BenchmarkReport(title="Test", results=results)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            report.save_json(path)
            loaded = BenchmarkReport.load_json(path)

            assert loaded.title == "Test"
            assert len(loaded.results) == 1
        finally:
            os.unlink(path)


# =============================================================================
# compare_results Tests
# =============================================================================

class TestCompareResults:
    """Test cases for compare_results function."""

    def test_compare_results_speedup(self):
        baseline = [
            BenchmarkResult(
                name="model", batch_size=1, latency_ms=10.0,
                throughput=100.0, memory_mb=50.0, std_dev_ms=1.0
            ),
        ]
        comparison = [
            BenchmarkResult(
                name="model", batch_size=1, latency_ms=5.0,
                throughput=200.0, memory_mb=50.0, std_dev_ms=0.5
            ),
        ]

        result = compare_results(baseline, comparison)

        assert "model_batch1" in result
        assert result["model_batch1"]["speedup"] == 2.0
        assert result["model_batch1"]["is_improvement"] is True

    def test_compare_results_slowdown(self):
        baseline = [
            BenchmarkResult(
                name="model", batch_size=1, latency_ms=5.0,
                throughput=200.0, memory_mb=50.0, std_dev_ms=0.5
            ),
        ]
        comparison = [
            BenchmarkResult(
                name="model", batch_size=1, latency_ms=10.0,
                throughput=100.0, memory_mb=50.0, std_dev_ms=1.0
            ),
        ]

        result = compare_results(baseline, comparison)

        assert result["model_batch1"]["speedup"] == 0.5
        assert result["model_batch1"]["is_improvement"] is False


# =============================================================================
# Benchmark Models Tests
# =============================================================================

class TestBenchmarkModels:
    """Test cases for benchmark models."""

    def test_list_benchmark_models(self):
        models = list_benchmark_models()
        assert len(models) > 0
        assert all(isinstance(m, BenchmarkModelInfo) for m in models)

    def test_list_benchmark_models_by_category(self):
        vision_models = list_benchmark_models(category="vision")
        assert len(vision_models) > 0
        assert all(m.category == "vision" for m in vision_models)

    def test_get_benchmark_model_info(self):
        info = get_benchmark_model_info("resnet50")
        assert info.name == "resnet50"
        assert info.input_shape == [3, 224, 224]

    def test_get_benchmark_model_info_not_found(self):
        with pytest.raises(ValueError):
            get_benchmark_model_info("nonexistent_model")

    def test_get_benchmark_model(self):
        model = get_benchmark_model("mlp_small", num_classes=10)
        assert model is not None
        assert callable(model)


# =============================================================================
# timed Context Manager Tests
# =============================================================================

class TestTimedContextManager:
    """Test cases for timed context manager."""

    def test_timed_context(self, capsys):
        with timed("Test operation"):
            _ = sum(range(1000))

        captured = capsys.readouterr()
        assert "Test operation:" in captured.out
        assert "ms" in captured.out


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
