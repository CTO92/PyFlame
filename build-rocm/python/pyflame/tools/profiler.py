"""
PyFlame Profiler for performance analysis.

Provides operation timing, memory tracking, and performance visualization.
"""

import functools
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ProfileEvent:
    """A single profiling event."""

    name: str
    category: str
    start_time: float
    end_time: float
    thread_id: int = 0
    process_id: int = 0
    args: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def duration_us(self) -> float:
        """Duration in microseconds."""
        return (self.end_time - self.start_time) * 1_000_000

    def to_chrome_trace(self) -> Dict[str, Any]:
        """Convert to Chrome trace format.

        Returns:
            Dictionary in Chrome trace event format
        """
        return {
            "name": self.name,
            "cat": self.category,
            "ph": "X",  # Complete event
            "ts": self.start_time * 1_000_000,  # Microseconds
            "dur": self.duration_us,
            "pid": self.process_id,
            "tid": self.thread_id,
            "args": self.args,
        }


@dataclass
class ProfileResult:
    """Results from a profiling session.

    Attributes:
        events: List of profiling events
        total_time: Total time in seconds
        operation_stats: Statistics per operation
        memory_stats: Memory usage statistics
    """

    events: List[ProfileEvent] = field(default_factory=list)
    total_time: float = 0.0
    operation_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    memory_stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get a summary of the profiling results.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "PyFlame Profile Summary",
            "=" * 60,
            f"Total Time: {self.total_time * 1000:.2f} ms",
            f"Total Events: {len(self.events)}",
            "",
            "Operation Breakdown:",
            "-" * 40,
        ]

        # Sort by total time
        sorted_ops = sorted(
            self.operation_stats.items(),
            key=lambda x: x[1].get("total_ms", 0),
            reverse=True,
        )

        for op_name, stats in sorted_ops[:20]:  # Top 20
            lines.append(
                f"  {op_name:<25} "
                f"{stats.get('total_ms', 0):>8.2f} ms "
                f"({stats.get('count', 0):>4} calls)"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_chrome_trace(self) -> Dict[str, Any]:
        """Export to Chrome trace format.

        Returns:
            Dictionary suitable for Chrome tracing (chrome://tracing)
        """
        return {
            "traceEvents": [e.to_chrome_trace() for e in self.events],
            "displayTimeUnit": "ms",
            "metadata": {
                "total_time_ms": self.total_time * 1000,
                "event_count": len(self.events),
            },
        }

    def save_chrome_trace(self, path: str):
        """Save to Chrome trace JSON file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            json.dump(self.to_chrome_trace(), f, indent=2)

    def get_top_operations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N operations by time.

        Args:
            n: Number of operations to return

        Returns:
            List of operation statistics
        """
        sorted_ops = sorted(
            self.operation_stats.items(),
            key=lambda x: x[1].get("total_ms", 0),
            reverse=True,
        )
        return [{"name": name, **stats} for name, stats in sorted_ops[:n]]


class Profiler:
    """Performance profiler for PyFlame operations.

    Tracks operation timing, memory usage, and provides detailed
    performance analysis.

    Example:
        >>> profiler = Profiler()
        >>> with profiler.profile():
        ...     output = model(x)
        >>> print(profiler.results.summary())
    """

    def __init__(
        self,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        """Initialize profiler.

        Args:
            record_shapes: Record tensor shapes
            profile_memory: Track memory usage
            with_stack: Record call stacks (expensive)
        """
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        self._events: List[ProfileEvent] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._active: bool = False
        self._operation_times: Dict[str, List[float]] = defaultdict(list)
        self._memory_snapshots: List[Dict[str, Any]] = []

        self.results: Optional[ProfileResult] = None

    @contextmanager
    def profile(self):
        """Context manager for profiling.

        Example:
            >>> with profiler.profile():
            ...     result = model(x)
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def start(self):
        """Start profiling."""
        self._events = []
        self._operation_times = defaultdict(list)
        self._memory_snapshots = []
        self._start_time = time.perf_counter()
        self._active = True

        if self.profile_memory:
            self._record_memory("start")

    def stop(self):
        """Stop profiling and compute results."""
        self._end_time = time.perf_counter()
        self._active = False

        if self.profile_memory:
            self._record_memory("end")

        self._compute_results()

    def record_event(
        self,
        name: str,
        category: str = "op",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **kwargs,
    ):
        """Record a profiling event.

        Args:
            name: Event name
            category: Event category (e.g., "op", "memory", "io")
            start_time: Event start time (default: now)
            end_time: Event end time (default: now)
            **kwargs: Additional event arguments
        """
        if not self._active:
            return

        now = time.perf_counter()
        event = ProfileEvent(
            name=name,
            category=category,
            start_time=start_time or now,
            end_time=end_time or now,
            args=kwargs,
        )
        self._events.append(event)

    @contextmanager
    def record_operation(self, name: str, category: str = "op", **kwargs):
        """Context manager to record an operation's timing.

        Args:
            name: Operation name
            category: Event category
            **kwargs: Additional arguments

        Example:
            >>> with profiler.record_operation("matmul"):
            ...     result = pf.matmul(a, b)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            if self._active:
                self.record_event(
                    name=name,
                    category=category,
                    start_time=start,
                    end_time=end,
                    **kwargs,
                )
                self._operation_times[name].append(end - start)

    def _record_memory(self, label: str):
        """Record current memory usage.

        Args:
            label: Label for the snapshot
        """
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()

            self._memory_snapshots.append(
                {
                    "label": label,
                    "time": time.perf_counter(),
                    "rss_mb": mem_info.rss / (1024 * 1024),
                    "vms_mb": mem_info.vms / (1024 * 1024),
                }
            )
        except ImportError:
            # psutil not available
            self._memory_snapshots.append(
                {
                    "label": label,
                    "time": time.perf_counter(),
                    "rss_mb": 0,
                    "vms_mb": 0,
                }
            )

    def _compute_results(self):
        """Compute profiling results from collected data."""
        total_time = (self._end_time or 0) - (self._start_time or 0)

        # Compute operation statistics
        operation_stats = {}
        for op_name, times in self._operation_times.items():
            if times:
                operation_stats[op_name] = {
                    "count": len(times),
                    "total_ms": sum(times) * 1000,
                    "mean_ms": (sum(times) / len(times)) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }

        # Aggregate event statistics
        event_times: Dict[str, List[float]] = defaultdict(list)
        for event in self._events:
            event_times[event.name].append(event.duration_ms)

        for name, times in event_times.items():
            if name not in operation_stats:
                operation_stats[name] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "mean_ms": sum(times) / len(times) if times else 0,
                    "min_ms": min(times) if times else 0,
                    "max_ms": max(times) if times else 0,
                }

        # Memory statistics
        memory_stats = {}
        if self._memory_snapshots:
            start_mem = self._memory_snapshots[0]
            end_mem = self._memory_snapshots[-1]
            memory_stats = {
                "start_rss_mb": start_mem["rss_mb"],
                "end_rss_mb": end_mem["rss_mb"],
                "delta_rss_mb": end_mem["rss_mb"] - start_mem["rss_mb"],
                "peak_rss_mb": max(s["rss_mb"] for s in self._memory_snapshots),
            }

        self.results = ProfileResult(
            events=self._events.copy(),
            total_time=total_time,
            operation_stats=operation_stats,
            memory_stats=memory_stats,
        )


def profile(func: Optional[Callable] = None, **profiler_kwargs) -> Callable:
    """Decorator to profile a function.

    Can be used with or without arguments.

    Example:
        >>> @profile
        ... def train_step(model, x, y):
        ...     return model(x)

        >>> @profile(record_shapes=True)
        ... def inference(model, x):
        ...     return model(x)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            profiler = Profiler(**profiler_kwargs)
            with profiler.profile():
                result = fn(*args, **kwargs)

            # Attach results to function for inspection
            wrapper._last_profile = profiler.results
            return result

        wrapper._last_profile = None
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class ProfilerContext:
    """Global profiler context for implicit profiling.

    Example:
        >>> ProfilerContext.enable()
        >>> result = model(x)
        >>> ProfilerContext.disable()
        >>> print(ProfilerContext.results.summary())
    """

    _profiler: Optional[Profiler] = None
    _results: Optional[ProfileResult] = None

    @classmethod
    def enable(cls, **kwargs):
        """Enable global profiling."""
        cls._profiler = Profiler(**kwargs)
        cls._profiler.start()

    @classmethod
    def disable(cls) -> Optional[ProfileResult]:
        """Disable global profiling and return results."""
        if cls._profiler:
            cls._profiler.stop()
            cls._results = cls._profiler.results
            cls._profiler = None
            return cls._results
        return None

    @classmethod
    def is_active(cls) -> bool:
        """Check if profiling is active."""
        return cls._profiler is not None and cls._profiler._active

    @classmethod
    def record(cls, name: str, **kwargs):
        """Record an event to the global profiler."""
        if cls._profiler and cls._profiler._active:
            cls._profiler.record_event(name, **kwargs)

    @classmethod
    @contextmanager
    def record_operation(cls, name: str, **kwargs):
        """Record an operation in the global profiler."""
        if cls._profiler and cls._profiler._active:
            with cls._profiler.record_operation(name, **kwargs):
                yield
        else:
            yield

    @classmethod
    def get_results(cls) -> Optional[ProfileResult]:
        """Get the last profiling results."""
        return cls._results
