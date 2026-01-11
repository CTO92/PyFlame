"""
PyFlame Benchmarking Module.

Provides performance benchmarking and comparison tools.
"""

from .runner import BenchmarkRunner, BenchmarkConfig, benchmark
from .results import BenchmarkResult, BenchmarkReport, compare_results
from .models import get_benchmark_model, list_benchmark_models

__all__ = [
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    "benchmark",

    # Results
    "BenchmarkResult",
    "BenchmarkReport",
    "compare_results",

    # Models
    "get_benchmark_model",
    "list_benchmark_models",
]
