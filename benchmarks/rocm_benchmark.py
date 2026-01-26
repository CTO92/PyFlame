#!/usr/bin/env python3
"""
ROCm Performance Benchmark Suite for PyFlame

This benchmark suite measures the performance of various operations on AMD GPUs
and compares them with CPU performance where applicable.

Usage:
    python rocm_benchmark.py [--all] [--matmul] [--conv] [--activations] [--memory]

Examples:
    python rocm_benchmark.py --all              # Run all benchmarks
    python rocm_benchmark.py --matmul --conv    # Run only matmul and conv
    python rocm_benchmark.py --iterations 200   # Custom iteration count
"""

import argparse
import time
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    import pyflame as pf
except ImportError:
    print("Error: PyFlame not found. Please install PyFlame first.")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Stores benchmark results."""
    name: str
    size: str
    time_ms: float
    throughput: Optional[float] = None  # TFLOPS for compute, GB/s for memory
    unit: str = "ms"


class ROCmBenchmark:
    """ROCm performance benchmark suite."""

    def __init__(self, iterations: int = 100, warmup: int = 10):
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []

    def _time_kernel(self, fn, iterations: int = None) -> float:
        """Time a kernel execution, returning average time in milliseconds."""
        if iterations is None:
            iterations = self.iterations

        # Warmup
        for _ in range(self.warmup):
            fn()
        pf.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        pf.synchronize()
        elapsed = time.perf_counter() - start

        return (elapsed / iterations) * 1000  # Convert to ms

    def benchmark_matmul(self):
        """Benchmark matrix multiplication (GEMM)."""
        print("\n" + "=" * 60)
        print("Matrix Multiplication (GEMM) Benchmark")
        print("=" * 60)

        sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            # Non-square matrices
            (1024, 4096, 1024),
            (4096, 1024, 4096),
        ]

        for M, K, N in sizes:
            a = pf.randn(M, K)
            b = pf.randn(K, N)

            def fn():
                return pf.matmul(a, b)

            time_ms = self._time_kernel(fn)

            # Calculate TFLOPS (2*M*N*K FLOPs for GEMM)
            flops = 2 * M * N * K
            tflops = (flops / (time_ms / 1000)) / 1e12

            result = BenchmarkResult(
                name="GEMM",
                size=f"{M}x{K}x{N}",
                time_ms=time_ms,
                throughput=tflops,
                unit="TFLOPS"
            )
            self.results.append(result)

            print(f"  {result.size:>15}: {result.time_ms:8.3f} ms | {result.throughput:6.2f} TFLOPS")

    def benchmark_batched_matmul(self):
        """Benchmark batched matrix multiplication."""
        print("\n" + "=" * 60)
        print("Batched Matrix Multiplication Benchmark")
        print("=" * 60)

        configs = [
            (8, 512, 512, 512),
            (16, 512, 512, 512),
            (32, 256, 256, 256),
            (64, 128, 128, 128),
            (128, 64, 64, 64),
        ]

        for B, M, K, N in configs:
            a = pf.randn(B, M, K)
            b = pf.randn(B, K, N)

            def fn():
                return pf.bmm(a, b)

            time_ms = self._time_kernel(fn)

            flops = B * 2 * M * N * K
            tflops = (flops / (time_ms / 1000)) / 1e12

            result = BenchmarkResult(
                name="BatchedGEMM",
                size=f"{B}x{M}x{K}x{N}",
                time_ms=time_ms,
                throughput=tflops,
                unit="TFLOPS"
            )
            self.results.append(result)

            print(f"  {result.size:>20}: {result.time_ms:8.3f} ms | {result.throughput:6.2f} TFLOPS")

    def benchmark_conv2d(self):
        """Benchmark 2D convolution."""
        print("\n" + "=" * 60)
        print("2D Convolution Benchmark")
        print("=" * 60)

        # Common configurations from ResNet, VGG, etc.
        configs = [
            # (N, C_in, H, W, C_out, kH, kW, stride, padding) - ResNet-like
            (1, 3, 224, 224, 64, 7, 7, 2, 3),     # ResNet first layer
            (32, 64, 56, 56, 64, 3, 3, 1, 1),     # ResNet block
            (32, 64, 56, 56, 128, 3, 3, 2, 1),    # ResNet downsample
            (32, 128, 28, 28, 128, 3, 3, 1, 1),
            (32, 256, 14, 14, 256, 3, 3, 1, 1),
            (32, 512, 7, 7, 512, 3, 3, 1, 1),
            # VGG-like
            (32, 64, 224, 224, 64, 3, 3, 1, 1),
            (32, 512, 14, 14, 512, 3, 3, 1, 1),
        ]

        for N, C_in, H, W, C_out, kH, kW, stride, padding in configs:
            x = pf.randn(N, C_in, H, W)
            w = pf.randn(C_out, C_in, kH, kW)

            def fn():
                return pf.conv2d(x, w, stride=stride, padding=padding)

            time_ms = self._time_kernel(fn)

            # Calculate output size
            H_out = (H + 2 * padding - kH) // stride + 1
            W_out = (W + 2 * padding - kW) // stride + 1

            # FLOPs for conv2d: 2 * N * C_out * H_out * W_out * C_in * kH * kW
            flops = 2 * N * C_out * H_out * W_out * C_in * kH * kW
            tflops = (flops / (time_ms / 1000)) / 1e12

            result = BenchmarkResult(
                name="Conv2D",
                size=f"{N}x{C_in}x{H}x{W}→{C_out}x{kH}x{kW}",
                time_ms=time_ms,
                throughput=tflops,
                unit="TFLOPS"
            )
            self.results.append(result)

            print(f"  {result.size:>30}: {result.time_ms:8.3f} ms | {result.throughput:6.2f} TFLOPS")

    def benchmark_activations(self):
        """Benchmark activation functions."""
        print("\n" + "=" * 60)
        print("Activation Functions Benchmark")
        print("=" * 60)

        sizes = [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
        activations = ["relu", "sigmoid", "tanh", "gelu", "silu"]

        for size in sizes:
            size_mb = (size * 4) / (1024 * 1024)
            print(f"\n  Size: {size:,} elements ({size_mb:.1f} MB)")

            x = pf.randn(size)

            for act_name in activations:
                act_fn = getattr(pf, act_name)

                def fn():
                    return act_fn(x)

                time_ms = self._time_kernel(fn)

                # Memory bandwidth: read input + write output
                bytes_moved = size * 4 * 2  # float32 read + write
                gbps = (bytes_moved / (time_ms / 1000)) / 1e9

                result = BenchmarkResult(
                    name=act_name,
                    size=f"{size:,}",
                    time_ms=time_ms,
                    throughput=gbps,
                    unit="GB/s"
                )
                self.results.append(result)

                print(f"    {act_name:>10}: {time_ms:8.3f} ms | {gbps:7.1f} GB/s")

    def benchmark_softmax(self):
        """Benchmark softmax operation."""
        print("\n" + "=" * 60)
        print("Softmax Benchmark")
        print("=" * 60)

        configs = [
            (32, 1000),      # Classification
            (32, 10000),     # Large vocab
            (128, 512),      # Attention
            (32, 512, 512),  # Full attention matrix
        ]

        for shape in configs:
            x = pf.randn(*shape)

            def fn():
                return pf.softmax(x, dim=-1)

            time_ms = self._time_kernel(fn)

            numel = np.prod(shape)
            bytes_moved = numel * 4 * 2
            gbps = (bytes_moved / (time_ms / 1000)) / 1e9

            shape_str = "x".join(map(str, shape))
            result = BenchmarkResult(
                name="Softmax",
                size=shape_str,
                time_ms=time_ms,
                throughput=gbps,
                unit="GB/s"
            )
            self.results.append(result)

            print(f"  {shape_str:>20}: {time_ms:8.3f} ms | {gbps:7.1f} GB/s")

    def benchmark_reductions(self):
        """Benchmark reduction operations."""
        print("\n" + "=" * 60)
        print("Reduction Operations Benchmark")
        print("=" * 60)

        sizes = [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
        reductions = ["sum", "mean", "max", "min"]

        for size in sizes:
            size_mb = (size * 4) / (1024 * 1024)
            print(f"\n  Size: {size:,} elements ({size_mb:.1f} MB)")

            x = pf.randn(size)

            for red_name in reductions:
                red_fn = getattr(pf, red_name)

                def fn():
                    return red_fn(x)

                time_ms = self._time_kernel(fn)

                bytes_read = size * 4
                gbps = (bytes_read / (time_ms / 1000)) / 1e9

                result = BenchmarkResult(
                    name=red_name,
                    size=f"{size:,}",
                    time_ms=time_ms,
                    throughput=gbps,
                    unit="GB/s"
                )
                self.results.append(result)

                print(f"    {red_name:>10}: {time_ms:8.3f} ms | {gbps:7.1f} GB/s")

    def benchmark_elementwise(self):
        """Benchmark elementwise operations."""
        print("\n" + "=" * 60)
        print("Elementwise Operations Benchmark")
        print("=" * 60)

        size = 16 * 1024 * 1024  # 16M elements
        size_mb = (size * 4) / (1024 * 1024)
        print(f"  Size: {size:,} elements ({size_mb:.1f} MB)")

        a = pf.randn(size)
        b = pf.randn(size)

        ops = [
            ("add", lambda: a + b),
            ("sub", lambda: a - b),
            ("mul", lambda: a * b),
            ("div", lambda: a / (b.abs() + 0.1)),
        ]

        for op_name, op_fn in ops:
            time_ms = self._time_kernel(op_fn)

            # Read 2 inputs, write 1 output
            bytes_moved = size * 4 * 3
            gbps = (bytes_moved / (time_ms / 1000)) / 1e9

            result = BenchmarkResult(
                name=op_name,
                size=f"{size:,}",
                time_ms=time_ms,
                throughput=gbps,
                unit="GB/s"
            )
            self.results.append(result)

            print(f"    {op_name:>10}: {time_ms:8.3f} ms | {gbps:7.1f} GB/s")

    def benchmark_memory_transfer(self):
        """Benchmark host-device memory transfers."""
        print("\n" + "=" * 60)
        print("Memory Transfer Benchmark")
        print("=" * 60)

        sizes_mb = [1, 4, 16, 64, 256, 1024]

        for size_mb in sizes_mb:
            size = size_mb * 1024 * 1024 // 4  # Number of floats

            # Host to device (create tensor from numpy)
            np_data = np.random.randn(size).astype(np.float32)

            def h2d():
                return pf.tensor(np_data)

            h2d_time = self._time_kernel(h2d, iterations=20)
            h2d_gbps = (size * 4 / (h2d_time / 1000)) / 1e9

            # Device to host
            gpu_tensor = pf.randn(size)

            def d2h():
                return gpu_tensor.numpy()

            d2h_time = self._time_kernel(d2h, iterations=20)
            d2h_gbps = (size * 4 / (d2h_time / 1000)) / 1e9

            print(f"  {size_mb:>4} MB  H→D: {h2d_time:8.3f} ms ({h2d_gbps:6.1f} GB/s) | "
                  f"D→H: {d2h_time:8.3f} ms ({d2h_gbps:6.1f} GB/s)")

            self.results.append(BenchmarkResult(
                name="H2D",
                size=f"{size_mb}MB",
                time_ms=h2d_time,
                throughput=h2d_gbps,
                unit="GB/s"
            ))
            self.results.append(BenchmarkResult(
                name="D2H",
                size=f"{size_mb}MB",
                time_ms=d2h_time,
                throughput=d2h_gbps,
                unit="GB/s"
            ))

    def benchmark_pooling(self):
        """Benchmark pooling operations."""
        print("\n" + "=" * 60)
        print("Pooling Operations Benchmark")
        print("=" * 60)

        configs = [
            (32, 64, 112, 112, 2, 2),   # After first conv
            (32, 128, 56, 56, 2, 2),
            (32, 256, 28, 28, 2, 2),
            (32, 512, 14, 14, 2, 2),
        ]

        for N, C, H, W, kH, kW in configs:
            x = pf.randn(N, C, H, W)

            def max_pool_fn():
                return pf.max_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))

            def avg_pool_fn():
                return pf.avg_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))

            max_time = self._time_kernel(max_pool_fn)
            avg_time = self._time_kernel(avg_pool_fn)

            input_size = f"{N}x{C}x{H}x{W}"
            print(f"  {input_size:>15} k={kH}x{kW}  MaxPool: {max_time:6.3f} ms | "
                  f"AvgPool: {avg_time:6.3f} ms")

            self.results.append(BenchmarkResult(
                name="MaxPool2D",
                size=input_size,
                time_ms=max_time
            ))
            self.results.append(BenchmarkResult(
                name="AvgPool2D",
                size=input_size,
                time_ms=avg_time
            ))

    def benchmark_batch_norm(self):
        """Benchmark batch normalization."""
        print("\n" + "=" * 60)
        print("Batch Normalization Benchmark")
        print("=" * 60)

        configs = [
            (32, 64, 56, 56),
            (32, 128, 28, 28),
            (32, 256, 14, 14),
            (32, 512, 7, 7),
            (64, 256, 14, 14),
        ]

        for N, C, H, W in configs:
            x = pf.randn(N, C, H, W)
            gamma = pf.randn(C)
            beta = pf.randn(C)
            mean = pf.randn(C)
            var = pf.randn(C).abs() + 0.1

            def fn():
                return pf.batch_norm(x, mean, var, gamma, beta, training=False)

            time_ms = self._time_kernel(fn)

            numel = N * C * H * W
            bytes_moved = numel * 4 * 2  # Read input, write output
            gbps = (bytes_moved / (time_ms / 1000)) / 1e9

            shape_str = f"{N}x{C}x{H}x{W}"
            print(f"  {shape_str:>15}: {time_ms:8.3f} ms | {gbps:7.1f} GB/s")

            self.results.append(BenchmarkResult(
                name="BatchNorm",
                size=shape_str,
                time_ms=time_ms,
                throughput=gbps,
                unit="GB/s"
            ))

    def benchmark_mlp_forward(self):
        """Benchmark full MLP forward pass."""
        print("\n" + "=" * 60)
        print("MLP Forward Pass Benchmark")
        print("=" * 60)

        configs = [
            # (batch, input_dim, hidden_dims, output_dim)
            (32, 784, [256, 128], 10),       # MNIST-like
            (32, 1024, [512, 256], 100),     # Medium
            (64, 2048, [1024, 512, 256], 1000),  # Large
        ]

        for batch, in_dim, hidden_dims, out_dim in configs:
            # Build MLP layers
            dims = [in_dim] + hidden_dims + [out_dim]
            weights = []
            biases = []

            for i in range(len(dims) - 1):
                w = pf.randn(dims[i], dims[i+1]) * 0.01
                b = pf.randn(dims[i+1])
                weights.append(w)
                biases.append(b)

            x = pf.randn(batch, in_dim)

            def mlp_forward():
                h = x
                for i, (w, b) in enumerate(zip(weights, biases)):
                    h = pf.matmul(h, w) + b
                    if i < len(weights) - 1:  # No activation on last layer
                        h = pf.relu(h)
                return h

            time_ms = self._time_kernel(mlp_forward)

            # Calculate total FLOPs
            total_flops = 0
            for i in range(len(dims) - 1):
                total_flops += 2 * batch * dims[i] * dims[i+1]  # matmul
                total_flops += batch * dims[i+1]  # bias
            tflops = (total_flops / (time_ms / 1000)) / 1e12

            arch_str = f"{in_dim}→{'-'.join(map(str, hidden_dims))}→{out_dim}"
            print(f"  batch={batch:>3} {arch_str:>25}: {time_ms:8.3f} ms | {tflops:6.3f} TFLOPS")

            self.results.append(BenchmarkResult(
                name="MLP",
                size=f"b{batch}_{arch_str}",
                time_ms=time_ms,
                throughput=tflops,
                unit="TFLOPS"
            ))

    def benchmark_attention(self):
        """Benchmark attention mechanism components."""
        print("\n" + "=" * 60)
        print("Attention Mechanism Benchmark")
        print("=" * 60)

        configs = [
            # (batch, seq_len, d_model, n_heads)
            (8, 512, 512, 8),
            (16, 256, 768, 12),
            (4, 1024, 1024, 16),
            (2, 2048, 768, 12),
        ]

        for batch, seq_len, d_model, n_heads in configs:
            d_head = d_model // n_heads

            # Query, Key, Value projections
            Q = pf.randn(batch, n_heads, seq_len, d_head)
            K = pf.randn(batch, n_heads, seq_len, d_head)
            V = pf.randn(batch, n_heads, seq_len, d_head)

            def attention():
                # Q @ K^T
                scores = pf.matmul(Q, K.transpose(-2, -1))
                # Scale
                scores = scores / (d_head ** 0.5)
                # Softmax
                attn = pf.softmax(scores, dim=-1)
                # Attention @ V
                return pf.matmul(attn, V)

            time_ms = self._time_kernel(attention)

            # FLOPs: 2 * batch * n_heads * (2 * seq^2 * d_head)
            flops = 2 * batch * n_heads * 2 * seq_len * seq_len * d_head
            tflops = (flops / (time_ms / 1000)) / 1e12

            config_str = f"b{batch}_s{seq_len}_d{d_model}_h{n_heads}"
            print(f"  {config_str:>25}: {time_ms:8.3f} ms | {tflops:6.3f} TFLOPS")

            self.results.append(BenchmarkResult(
                name="Attention",
                size=config_str,
                time_ms=time_ms,
                throughput=tflops,
                unit="TFLOPS"
            ))

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # Group by operation type
        categories = {}
        for r in self.results:
            if r.name not in categories:
                categories[r.name] = []
            categories[r.name].append(r)

        for cat, results in categories.items():
            print(f"\n{cat}:")
            for r in results:
                if r.throughput:
                    print(f"  {r.size:>25}: {r.time_ms:8.3f} ms | {r.throughput:7.2f} {r.unit}")
                else:
                    print(f"  {r.size:>25}: {r.time_ms:8.3f} ms")

    def run_all(self):
        """Run all benchmarks."""
        self.benchmark_matmul()
        self.benchmark_batched_matmul()
        self.benchmark_conv2d()
        self.benchmark_activations()
        self.benchmark_softmax()
        self.benchmark_reductions()
        self.benchmark_elementwise()
        self.benchmark_pooling()
        self.benchmark_batch_norm()
        self.benchmark_memory_transfer()
        self.benchmark_mlp_forward()
        self.benchmark_attention()
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="ROCm Performance Benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--matmul", action="store_true", help="Run matmul benchmarks")
    parser.add_argument("--conv", action="store_true", help="Run convolution benchmarks")
    parser.add_argument("--activations", action="store_true", help="Run activation benchmarks")
    parser.add_argument("--reductions", action="store_true", help="Run reduction benchmarks")
    parser.add_argument("--elementwise", action="store_true", help="Run elementwise benchmarks")
    parser.add_argument("--memory", action="store_true", help="Run memory transfer benchmarks")
    parser.add_argument("--pooling", action="store_true", help="Run pooling benchmarks")
    parser.add_argument("--batchnorm", action="store_true", help="Run batch norm benchmarks")
    parser.add_argument("--mlp", action="store_true", help="Run MLP benchmarks")
    parser.add_argument("--attention", action="store_true", help="Run attention benchmarks")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")

    args = parser.parse_args()

    # Check ROCm availability
    if not pf.rocm_is_available():
        print("Error: ROCm not available")
        print("Please ensure:")
        print("  1. ROCm is installed")
        print("  2. A compatible AMD GPU is present")
        print("  3. PyFlame was built with -DPYFLAME_USE_ROCM=ON")
        sys.exit(1)

    # Set device and print info
    pf.set_device('rocm')
    info = pf.device_info()

    print("=" * 60)
    print("ROCm Performance Benchmark Suite")
    print("=" * 60)
    print(f"Device: {info['name']}")
    print(f"Architecture: {info.get('architecture', 'N/A')}")
    print(f"Memory: {info['total_memory'] / (1024**3):.1f} GB total, "
          f"{info['free_memory'] / (1024**3):.1f} GB free")
    print(f"Compute Units: {info.get('compute_units', 'N/A')}")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")

    benchmark = ROCmBenchmark(iterations=args.iterations, warmup=args.warmup)

    # If no specific benchmark selected, run all
    run_specific = any([
        args.matmul, args.conv, args.activations, args.reductions,
        args.elementwise, args.memory, args.pooling, args.batchnorm,
        args.mlp, args.attention
    ])

    if args.all or not run_specific:
        benchmark.run_all()
    else:
        if args.matmul:
            benchmark.benchmark_matmul()
            benchmark.benchmark_batched_matmul()
        if args.conv:
            benchmark.benchmark_conv2d()
        if args.activations:
            benchmark.benchmark_activations()
            benchmark.benchmark_softmax()
        if args.reductions:
            benchmark.benchmark_reductions()
        if args.elementwise:
            benchmark.benchmark_elementwise()
        if args.memory:
            benchmark.benchmark_memory_transfer()
        if args.pooling:
            benchmark.benchmark_pooling()
        if args.batchnorm:
            benchmark.benchmark_batch_norm()
        if args.mlp:
            benchmark.benchmark_mlp_forward()
        if args.attention:
            benchmark.benchmark_attention()

        benchmark.print_summary()


if __name__ == "__main__":
    main()
