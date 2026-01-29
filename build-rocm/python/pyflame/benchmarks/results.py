"""
Benchmark results and reporting for PyFlame.

Provides result analysis, comparison, and visualization.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BenchmarkResult:
    """Single benchmark result (re-exported from runner)."""

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


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report.

    Attributes:
        title: Report title
        results: List of benchmark results
        system_info: System information
        timestamp: Report timestamp
    """

    title: str
    results: List[BenchmarkResult]
    system_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()

        if not self.system_info:
            self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        import platform
        import sys

        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "processor": platform.processor(),
        }

        try:
            import psutil

            info["cpu_count"] = psutil.cpu_count()
            info["memory_total_gb"] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass

        return info

    def summary(self) -> str:
        """Generate report summary.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 80,
            f"Benchmark Report: {self.title}",
            f"Generated: {self.timestamp}",
            "=" * 80,
            "",
            "System Information:",
            "-" * 40,
        ]

        for key, value in self.system_info.items():
            lines.append(f"  {key}: {value}")

        lines.extend(
            [
                "",
                "Results Summary:",
                "-" * 40,
                f"{'Model':<25} {'Batch':<8} {'Latency (ms)':<15} {'Throughput':<15}",
                "-" * 70,
            ]
        )

        for r in self.results:
            lines.append(
                f"{r.name:<25} {r.batch_size:<8} "
                f"{r.latency_ms:>8.2f} +/- {r.std_dev_ms:<4.1f} "
                f"{r.throughput:>10.1f}/s"
            )

        lines.extend(
            [
                "",
                "=" * 80,
            ]
        )

        if self.notes:
            lines.extend(["Notes:", self.notes, ""])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "results": [
                {
                    "name": r.name,
                    "batch_size": r.batch_size,
                    "latency_ms": r.latency_ms,
                    "throughput": r.throughput,
                    "memory_mb": r.memory_mb,
                    "std_dev_ms": r.std_dev_ms,
                    "min_ms": r.min_ms,
                    "max_ms": r.max_ms,
                    "percentile_95_ms": r.percentile_95_ms,
                    "percentile_99_ms": r.percentile_99_ms,
                }
                for r in self.results
            ],
            "notes": self.notes,
        }

    def save_json(self, path: str):
        """Save report to JSON file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_html(self, path: str):
        """Save report as HTML.

        Args:
            path: Output file path
        """
        html = self._generate_html()
        with open(path, "w") as f:
            f.write(html)

    def _generate_html(self) -> str:
        """Generate HTML report."""
        rows = ""
        for r in self.results:
            rows += f"""
            <tr>
                <td>{r.name}</td>
                <td>{r.batch_size}</td>
                <td>{r.latency_ms:.2f}</td>
                <td>{r.std_dev_ms:.2f}</td>
                <td>{r.throughput:.1f}</td>
                <td>{r.memory_mb:.1f}</td>
            </tr>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .system-info {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .system-info p {{
            margin: 5px 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <p class="timestamp">Generated: {self.timestamp}</p>

        <h2>System Information</h2>
        <div class="system-info">
            {"".join(f"<p><strong>{k}:</strong> {v}</p>" for k, v in self.system_info.items())}
        </div>

        <h2>Benchmark Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Batch Size</th>
                    <th>Latency (ms)</th>
                    <th>Std Dev (ms)</th>
                    <th>Throughput (/s)</th>
                    <th>Memory (MB)</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>

        {f"<h2>Notes</h2><p>{self.notes}</p>" if self.notes else ""}
    </div>
</body>
</html>"""

    @classmethod
    def load_json(cls, path: str) -> "BenchmarkReport":
        """Load report from JSON file.

        Args:
            path: Input file path

        Returns:
            BenchmarkReport instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        results = [BenchmarkResult(**r) for r in data.get("results", [])]

        return cls(
            title=data.get("title", "Benchmark Report"),
            results=results,
            system_info=data.get("system_info", {}),
            timestamp=data.get("timestamp", ""),
            notes=data.get("notes", ""),
        )


def compare_results(
    baseline: List[BenchmarkResult],
    comparison: List[BenchmarkResult],
    metric: str = "latency_ms",
) -> Dict[str, Dict[str, Any]]:
    """Compare two sets of benchmark results.

    Args:
        baseline: Baseline benchmark results
        comparison: Comparison benchmark results
        metric: Metric to compare ("latency_ms", "throughput", "memory_mb")

    Returns:
        Dictionary with comparison for each benchmark

    Example:
        >>> comparison = compare_results(old_results, new_results)
        >>> print(comparison["resnet50"]["speedup"])
    """
    comparisons = {}

    # Index comparison results by name and batch size
    comp_index = {(r.name, r.batch_size): r for r in comparison}

    for base_result in baseline:
        key = (base_result.name, base_result.batch_size)
        comp_result = comp_index.get(key)

        if comp_result is None:
            continue

        base_value = getattr(base_result, metric, 0)
        comp_value = getattr(comp_result, metric, 0)

        if base_value == 0:
            ratio = float("inf") if comp_value > 0 else 1.0
        else:
            ratio = comp_value / base_value

        # For latency, lower is better
        if metric == "latency_ms":
            speedup = 1.0 / ratio if ratio > 0 else float("inf")
            improvement_pct = (1 - ratio) * 100
        else:
            # For throughput, higher is better
            speedup = ratio
            improvement_pct = (ratio - 1) * 100

        result_key = f"{base_result.name}_batch{base_result.batch_size}"
        comparisons[result_key] = {
            "baseline": base_value,
            "comparison": comp_value,
            "ratio": ratio,
            "speedup": speedup,
            "improvement_percent": improvement_pct,
            "is_improvement": improvement_pct > 0,
        }

    return comparisons


def print_comparison(
    comparisons: Dict[str, Dict[str, Any]],
    metric_name: str = "Latency",
):
    """Print comparison results.

    Args:
        comparisons: Comparison dictionary from compare_results
        metric_name: Name of the compared metric
    """
    print()
    print("=" * 80)
    print(f"Benchmark Comparison ({metric_name})")
    print("=" * 80)
    print(
        f"{'Benchmark':<30} {'Baseline':<12} {'New':<12} "
        f"{'Change':<12} {'Speedup':<10}"
    )
    print("-" * 80)

    for name, data in comparisons.items():
        change = data["improvement_percent"]
        change_str = f"{change:+.1f}%"
        if data["is_improvement"]:
            change_str = f"\033[32m{change_str}\033[0m"  # Green
        else:
            change_str = f"\033[31m{change_str}\033[0m"  # Red

        print(
            f"{name:<30} {data['baseline']:<12.2f} {data['comparison']:<12.2f} "
            f"{change_str:<20} {data['speedup']:<10.2f}x"
        )

    print("=" * 80)
    print()
