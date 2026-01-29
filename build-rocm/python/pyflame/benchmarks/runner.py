"""
Benchmark runner for PyFlame.

Provides systematic benchmarking of models and operations.
"""

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class nullcontext:
    """Context manager that does nothing (fallback for older Python versions)."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        warmup_iterations: Iterations for warmup
        benchmark_iterations: Iterations for measurement
        batch_sizes: Batch sizes to test
        precision: Data precision
        profile_memory: Track memory usage
        device: Device to run on
    """

    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 64])
    precision: str = "fp32"
    profile_memory: bool = True
    device: str = "cpu"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes:
        name: Benchmark name
        batch_size: Batch size used
        latency_ms: Average latency in milliseconds
        throughput: Throughput (samples/second)
        memory_mb: Memory usage in MB
        std_dev_ms: Standard deviation of latency
        min_ms: Minimum latency
        max_ms: Maximum latency
        percentile_95_ms: 95th percentile latency
        percentile_99_ms: 99th percentile latency
    """

    name: str
    batch_size: int
    latency_ms: float
    throughput: float
    memory_mb: float
    std_dev_ms: float
    min_ms: float = 0.0
    max_ms: float = 0.0
    percentile_95_ms: float = 0.0
    percentile_99_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "memory_mb": self.memory_mb,
            "std_dev_ms": self.std_dev_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "percentile_95_ms": self.percentile_95_ms,
            "percentile_99_ms": self.percentile_99_ms,
            "metadata": self.metadata,
        }


class BenchmarkRunner:
    """Run benchmarks on PyFlame models and operations.

    Example:
        >>> runner = BenchmarkRunner()
        >>> results = runner.run_model_benchmark(
        ...     "resnet50",
        ...     model=resnet50(),
        ...     input_shape=[3, 224, 224]
        ... )
        >>> runner.print_results(results)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []

    def run_model_benchmark(
        self,
        name: str,
        model,
        input_shape: List[int],
        batch_sizes: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """Benchmark a model across different batch sizes.

        Args:
            name: Benchmark name
            model: Model to benchmark
            input_shape: Input shape (without batch dimension)
            batch_sizes: Batch sizes to test

        Returns:
            List of benchmark results
        """
        batch_sizes = batch_sizes or self.config.batch_sizes
        results = []

        # Set model to eval mode
        if hasattr(model, "eval"):
            model.eval()

        for batch_size in batch_sizes:
            result = self._benchmark_batch_size(name, model, input_shape, batch_size)
            results.append(result)
            self.results.append(result)

        return results

    def _benchmark_batch_size(
        self,
        name: str,
        model,
        input_shape: List[int],
        batch_size: int,
    ) -> BenchmarkResult:
        """Benchmark model at specific batch size.

        Args:
            name: Benchmark name
            model: Model to benchmark
            input_shape: Input shape
            batch_size: Batch size

        Returns:
            Benchmark result
        """
        try:
            import pyflame as pf

            full_shape = [batch_size] + input_shape
            x = pf.randn(full_shape)

            # Warmup
            no_grad = pf.no_grad() if hasattr(pf, "no_grad") else nullcontext()
            with no_grad:
                for _ in range(self.config.warmup_iterations):
                    _ = model(x)

            # Benchmark
            times = []
            with no_grad:
                for _ in range(self.config.benchmark_iterations):
                    start = time.perf_counter()
                    _ = model(x)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms

        except ImportError:
            # Fallback for when pyflame is not available
            import numpy as np

            full_shape = [batch_size] + input_shape
            x = np.random.randn(*full_shape).astype(np.float32)

            # Warmup
            for _ in range(self.config.warmup_iterations):
                _ = model(x)

            # Benchmark
            times = []
            for _ in range(self.config.benchmark_iterations):
                start = time.perf_counter()
                _ = model(x)
                end = time.perf_counter()
                times.append((end - start) * 1000)

        # Calculate statistics
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)

        sorted_times = sorted(times)
        n = len(sorted_times)
        p95_idx = min(int((n - 1) * 0.95), n - 1) if n > 0 else 0
        p99_idx = min(int((n - 1) * 0.99), n - 1) if n > 0 else 0
        p95 = sorted_times[p95_idx] if n > 0 else max_time
        p99 = sorted_times[p99_idx] if n > 0 else max_time

        throughput = batch_size / (avg_time / 1000)  # samples/sec

        # Memory profiling
        memory_mb = 0.0
        if self.config.profile_memory:
            memory_mb = self._get_memory_usage()

        return BenchmarkResult(
            name=name,
            batch_size=batch_size,
            latency_ms=avg_time,
            throughput=throughput,
            memory_mb=memory_mb,
            std_dev_ms=std_dev,
            min_ms=min_time,
            max_ms=max_time,
            percentile_95_ms=p95,
            percentile_99_ms=p99,
            metadata={
                "input_shape": input_shape,
                "iterations": self.config.benchmark_iterations,
                "precision": self.config.precision,
            },
        )

    def run_operation_benchmark(
        self,
        name: str,
        operation: Callable,
        input_shapes: List[List[int]],
    ) -> BenchmarkResult:
        """Benchmark a single operation.

        Args:
            name: Benchmark name
            operation: Operation function to benchmark
            input_shapes: Shapes of input tensors

        Returns:
            Benchmark result
        """
        try:
            import pyflame as pf

            inputs = [pf.randn(shape) for shape in input_shapes]
        except ImportError:
            import numpy as np

            inputs = [
                np.random.randn(*shape).astype(np.float32) for shape in input_shapes
            ]

        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = operation(*inputs)

        # Benchmark
        times = []
        for _ in range(self.config.benchmark_iterations):
            start = time.perf_counter()
            _ = operation(*inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

        result = BenchmarkResult(
            name=name,
            batch_size=input_shapes[0][0] if input_shapes else 1,
            latency_ms=avg_time,
            throughput=1000 / avg_time if avg_time > 0 else 0,
            memory_mb=self._get_memory_usage() if self.config.profile_memory else 0,
            std_dev_ms=std_dev,
            min_ms=min(times),
            max_ms=max(times),
            metadata={"input_shapes": input_shapes},
        )

        self.results.append(result)
        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def print_results(self, results: Optional[List[BenchmarkResult]] = None):
        """Print benchmark results in table format.

        Args:
            results: Results to print (default: all results)
        """
        results = results or self.results

        if not results:
            print("No benchmark results to display")
            return

        print()
        print("=" * 90)
        print("PyFlame Benchmark Results")
        print("=" * 90)
        print(
            f"{'Model':<25} {'Batch':<8} {'Latency (ms)':<18} "
            f"{'Throughput':<15} {'Memory (MB)':<12}"
        )
        print("-" * 90)

        for r in results:
            print(
                f"{r.name:<25} {r.batch_size:<8} "
                f"{r.latency_ms:>8.2f} +/- {r.std_dev_ms:<6.2f}  "
                f"{r.throughput:>10.1f}/s   {r.memory_mb:>10.1f}"
            )

        print("=" * 90)
        print()

    def export_json(self, path: str):
        """Export results to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "config": {
                "warmup_iterations": self.config.warmup_iterations,
                "benchmark_iterations": self.config.benchmark_iterations,
                "precision": self.config.precision,
            },
            "results": [r.to_dict() for r in self.results],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, path: str):
        """Export results to CSV file.

        Args:
            path: Output file path
        """
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "batch_size",
                    "latency_ms",
                    "std_dev_ms",
                    "throughput",
                    "memory_mb",
                    "min_ms",
                    "max_ms",
                    "percentile_95_ms",
                    "percentile_99_ms",
                ]
            )
            for r in self.results:
                writer.writerow(
                    [
                        r.name,
                        r.batch_size,
                        r.latency_ms,
                        r.std_dev_ms,
                        r.throughput,
                        r.memory_mb,
                        r.min_ms,
                        r.max_ms,
                        r.percentile_95_ms,
                        r.percentile_99_ms,
                    ]
                )

    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()


def benchmark(
    model,
    input_shape: List[int],
    batch_sizes: Optional[List[int]] = None,
    iterations: int = 100,
    print_results: bool = True,
) -> List[BenchmarkResult]:
    """Quick benchmark function for a model.

    Convenience function for running benchmarks.

    Args:
        model: Model to benchmark
        input_shape: Input shape (without batch dimension)
        batch_sizes: Batch sizes to test
        iterations: Number of benchmark iterations
        print_results: Print results after benchmarking

    Returns:
        List of benchmark results

    Example:
        >>> results = benchmark(model, [3, 224, 224])
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]
    config = BenchmarkConfig(
        benchmark_iterations=iterations,
        batch_sizes=batch_sizes,
    )
    runner = BenchmarkRunner(config)
    results = runner.run_model_benchmark("model", model, input_shape)

    if print_results:
        runner.print_results(results)

    return results


@contextmanager
def timed(name: str = "Operation"):
    """Context manager for timing operations.

    Example:
        >>> with timed("Forward pass"):
        ...     output = model(x)
        Forward pass: 12.34 ms
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.2f} ms")
