"""
Tests for PyFlame Phase 4 Developer Tools Module.

Tests debugger, profiler, and visualization tools.
"""

import pytest
import sys
import os
import tempfile

# Add Python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pyflame.tools.debugger import (
    PyFlameDebugger,
    Breakpoint,
    BreakpointType,
    TensorInspector,
    set_breakpoint,
    clear_breakpoints,
)
from pyflame.tools.profiler import (
    Profiler,
    ProfileResult,
    ProfileEvent,
    profile,
    ProfilerContext,
)
from pyflame.tools.visualization import (
    GraphVisualizer,
    visualize_graph,
)


# =============================================================================
# Debugger Tests
# =============================================================================

class TestPyFlameDebugger:
    """Test cases for PyFlameDebugger."""

    def test_debugger_initialization(self):
        debugger = PyFlameDebugger()
        assert debugger.breakpoints == {}
        assert debugger.watch_tensors == {}
        assert not debugger._active

    def test_debugger_context_manager(self):
        debugger = PyFlameDebugger()
        assert not debugger._active

        with debugger:
            assert debugger._active
            assert PyFlameDebugger.get_active() is debugger

        assert not debugger._active
        assert PyFlameDebugger.get_active() is None

    def test_set_breakpoint(self):
        debugger = PyFlameDebugger()
        bp_id = debugger.set_breakpoint(op="matmul")

        assert bp_id > 0
        assert bp_id in debugger.breakpoints

        bp = debugger.breakpoints[bp_id]
        assert bp.type == BreakpointType.OPERATION
        assert bp.op_name == "matmul"
        assert bp.enabled

    def test_remove_breakpoint(self):
        debugger = PyFlameDebugger()
        bp_id = debugger.set_breakpoint(op="relu")

        assert bp_id in debugger.breakpoints
        result = debugger.remove_breakpoint(bp_id)
        assert result is True
        assert bp_id not in debugger.breakpoints

        # Removing non-existent breakpoint
        result = debugger.remove_breakpoint(999)
        assert result is False

    def test_clear_breakpoints(self):
        debugger = PyFlameDebugger()
        debugger.set_breakpoint(op="relu")
        debugger.set_breakpoint(op="sigmoid")

        assert len(debugger.breakpoints) == 2
        debugger.clear_breakpoints()
        assert len(debugger.breakpoints) == 0

    def test_enable_disable_breakpoint(self):
        debugger = PyFlameDebugger()
        bp_id = debugger.set_breakpoint(op="conv")

        assert debugger.breakpoints[bp_id].enabled is True
        debugger.enable_breakpoint(bp_id, False)
        assert debugger.breakpoints[bp_id].enabled is False
        debugger.enable_breakpoint(bp_id, True)
        assert debugger.breakpoints[bp_id].enabled is True

    def test_watch_tensor(self):
        debugger = PyFlameDebugger()

        class MockTensor:
            pass

        tensor = MockTensor()
        debugger.watch("test_tensor", tensor)

        assert "test_tensor" in debugger.watch_tensors
        assert debugger.watch_tensors["test_tensor"] is tensor

        debugger.unwatch("test_tensor")
        assert "test_tensor" not in debugger.watch_tensors

    def test_global_breakpoint_functions(self):
        clear_breakpoints()

        bp_id = set_breakpoint(op="softmax")
        assert bp_id > 0

        clear_breakpoints()


class TestTensorInspector:
    """Test cases for TensorInspector."""

    def test_inspect_mock_tensor(self):
        class MockTensor:
            shape = [2, 3, 4]
            dtype = "float32"
            numel = 24

            def is_evaluated(self):
                return False

        tensor = MockTensor()
        info = TensorInspector.inspect(tensor)

        assert info["shape"] == [2, 3, 4]
        assert info["dtype"] == "float32"
        assert info["is_evaluated"] is False
        assert info["numel"] == 24


class TestBreakpoint:
    """Test cases for Breakpoint class."""

    def test_breakpoint_creation(self):
        bp = Breakpoint(
            id=1,
            type=BreakpointType.OPERATION,
            op_name="matmul",
        )
        assert bp.id == 1
        assert bp.type == BreakpointType.OPERATION
        assert bp.enabled is True
        assert bp.hit_count == 0

    def test_breakpoint_check_disabled(self):
        bp = Breakpoint(id=1, type=BreakpointType.CUSTOM, enabled=False)
        assert bp.check({}) is False

    def test_breakpoint_check_with_condition(self):
        bp = Breakpoint(
            id=1,
            type=BreakpointType.CUSTOM,
            condition=lambda ctx: ctx.get("value", 0) > 10,
        )

        assert bp.check({"value": 5}) is False
        assert bp.check({"value": 15}) is True


# =============================================================================
# Profiler Tests
# =============================================================================

class TestProfiler:
    """Test cases for Profiler."""

    def test_profiler_initialization(self):
        profiler = Profiler()
        assert profiler.record_shapes is True
        assert profiler.profile_memory is True
        assert profiler._active is False

    def test_profiler_context_manager(self):
        profiler = Profiler()

        with profiler.profile():
            assert profiler._active is True
            # Simulate some work
            _ = sum(range(1000))

        assert profiler._active is False
        assert profiler.results is not None
        assert profiler.results.total_time > 0

    def test_profiler_record_event(self):
        profiler = Profiler()
        profiler.start()

        profiler.record_event("test_event", category="test")
        profiler.record_event("another_event", category="test")

        profiler.stop()

        assert len(profiler._events) == 2

    def test_profiler_record_operation(self):
        profiler = Profiler()
        profiler.start()

        with profiler.record_operation("test_op"):
            _ = sum(range(1000))

        profiler.stop()

        assert "test_op" in profiler.results.operation_stats

    def test_profile_result_summary(self):
        profiler = Profiler()

        with profiler.profile():
            _ = sum(range(1000))

        summary = profiler.results.summary()
        assert "PyFlame Profile Summary" in summary
        assert "Total Time" in summary


class TestProfileEvent:
    """Test cases for ProfileEvent."""

    def test_profile_event_duration(self):
        event = ProfileEvent(
            name="test",
            category="op",
            start_time=0.0,
            end_time=0.001,  # 1 ms
        )

        assert event.duration_ms == 1.0
        assert event.duration_us == 1000.0

    def test_profile_event_to_chrome_trace(self):
        event = ProfileEvent(
            name="test",
            category="op",
            start_time=0.001,
            end_time=0.002,
        )

        trace = event.to_chrome_trace()
        assert trace["name"] == "test"
        assert trace["cat"] == "op"
        assert trace["ph"] == "X"
        assert "ts" in trace
        assert "dur" in trace


class TestProfileDecorator:
    """Test cases for @profile decorator."""

    def test_profile_decorator(self):
        @profile
        def my_function(x):
            return x * 2

        result = my_function(10)
        assert result == 20
        assert my_function._last_profile is not None


class TestProfilerContext:
    """Test cases for ProfilerContext."""

    def test_profiler_context_enable_disable(self):
        assert ProfilerContext.is_active() is False

        ProfilerContext.enable()
        assert ProfilerContext.is_active() is True

        result = ProfilerContext.disable()
        assert ProfilerContext.is_active() is False
        assert result is not None


# =============================================================================
# Visualization Tests
# =============================================================================

class TestGraphVisualizer:
    """Test cases for GraphVisualizer."""

    def test_visualizer_initialization(self):
        viz = GraphVisualizer()
        assert viz.graph is None
        assert viz.show_shapes is True
        assert viz.show_dtypes is True

    def test_visualizer_with_options(self):
        viz = GraphVisualizer(
            show_shapes=False,
            show_dtypes=False,
            max_nodes=100,
            rankdir="LR",
        )
        assert viz.show_shapes is False
        assert viz.max_nodes == 100
        assert viz.rankdir == "LR"

    def test_visualizer_op_category(self):
        viz = GraphVisualizer()

        assert viz._get_op_category("matmul") == "matmul"
        assert viz._get_op_category("relu") == "activation"
        assert viz._get_op_category("batchnorm") == "norm"
        assert viz._get_op_category("conv2d") == "conv"
        assert viz._get_op_category("input") == "input"
        assert viz._get_op_category("unknown_op") == "default"

    def test_visualizer_format_shape(self):
        viz = GraphVisualizer()
        assert viz._format_shape([2, 3, 4]) == "[2, 3, 4]"
        assert viz._format_shape([10]) == "[10]"
        assert viz._format_shape([]) == "[]"

    def test_visualizer_to_dot_empty(self):
        viz = GraphVisualizer()
        dot = viz.to_dot()

        assert "digraph PyFlameGraph" in dot
        assert "rankdir=TB" in dot


class TestVisualizeGraph:
    """Test cases for visualize_graph function."""

    def test_visualize_graph_dot_format(self):
        result = visualize_graph(None, format="dot")
        assert "digraph" in result

    def test_visualize_graph_invalid_format(self):
        with pytest.raises(ValueError):
            visualize_graph(None, format="invalid")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
