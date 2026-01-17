"""
PyFlame Debugger for inspecting tensors and computation graphs.

Provides breakpoints, tensor inspection, and step-by-step execution.
"""

import functools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class BreakpointType(Enum):
    """Type of breakpoint."""

    OPERATION = "operation"
    TENSOR_SHAPE = "tensor_shape"
    TENSOR_VALUE = "tensor_value"
    GRADIENT = "gradient"
    CUSTOM = "custom"


@dataclass
class Breakpoint:
    """A breakpoint in PyFlame execution.

    Attributes:
        id: Unique breakpoint identifier
        type: Type of breakpoint
        condition: Optional condition function
        enabled: Whether breakpoint is active
        hit_count: Number of times breakpoint was hit
        callback: Optional callback when hit
    """

    id: int
    type: BreakpointType
    condition: Optional[Callable[..., bool]] = None
    enabled: bool = True
    hit_count: int = 0
    callback: Optional[Callable[..., None]] = None
    op_name: Optional[str] = None
    tensor_name: Optional[str] = None

    def check(self, context: Dict[str, Any]) -> bool:
        """Check if breakpoint should trigger."""
        if not self.enabled:
            return False

        if self.condition is not None:
            try:
                return self.condition(context)
            except Exception:
                return False

        return True


class TensorInspector:
    """Utility for inspecting tensor properties."""

    @staticmethod
    def inspect(tensor) -> Dict[str, Any]:
        """Get detailed information about a tensor.

        Args:
            tensor: PyFlame tensor to inspect

        Returns:
            Dictionary with tensor properties
        """
        info = {
            "shape": list(tensor.shape) if hasattr(tensor, "shape") else None,
            "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else None,
            "is_evaluated": (
                tensor.is_evaluated() if hasattr(tensor, "is_evaluated") else None
            ),
            "numel": tensor.numel() if hasattr(tensor, "numel") else None,
        }

        # Add statistics if tensor is evaluated
        if info["is_evaluated"]:
            try:
                data = tensor.numpy()
                info["stats"] = {
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                }
                import numpy as np
                info["has_nan"] = bool(np.isnan(data).any())
                info["has_inf"] = bool(np.isinf(data).any())
            except Exception:
                info["stats"] = None

        return info

    @staticmethod
    def compare(
        tensor1, tensor2, rtol: float = 1e-5, atol: float = 1e-8
    ) -> Dict[str, Any]:
        """Compare two tensors.

        Args:
            tensor1: First tensor
            tensor2: Second tensor
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Comparison results
        """
        try:
            import numpy as np

            arr1 = tensor1.numpy()
            arr2 = tensor2.numpy()

            return {
                "shapes_match": arr1.shape == arr2.shape,
                "all_close": bool(np.allclose(arr1, arr2, rtol=rtol, atol=atol)),
                "max_diff": (
                    float(np.abs(arr1 - arr2).max())
                    if arr1.shape == arr2.shape
                    else None
                ),
                "mean_diff": (
                    float(np.abs(arr1 - arr2).mean())
                    if arr1.shape == arr2.shape
                    else None
                ),
            }
        except Exception as e:
            return {"error": str(e)}


class PyFlameDebugger:
    """Interactive debugger for PyFlame operations.

    Allows setting breakpoints on operations, inspecting tensors,
    and stepping through computation graphs.

    Example:
        >>> debugger = PyFlameDebugger()
        >>> debugger.set_breakpoint(op="matmul")
        >>> with debugger:
        ...     result = model(x)
    """

    _instance: Optional["PyFlameDebugger"] = None
    _breakpoint_counter: int = 0

    def __init__(self, verbose: bool = False):
        """Initialize debugger.

        Args:
            verbose: Print debug information
        """
        self.verbose = verbose
        self.breakpoints: Dict[int, Breakpoint] = {}
        self.watch_tensors: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.paused: bool = False
        self.step_mode: bool = False
        self._active: bool = False
        self._inspector = TensorInspector()

    def __enter__(self):
        """Enter debug context."""
        self._active = True
        PyFlameDebugger._instance = self
        if self.verbose:
            print("[PyFlame Debugger] Started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit debug context."""
        self._active = False
        PyFlameDebugger._instance = None
        if self.verbose:
            print("[PyFlame Debugger] Stopped")
        return False

    @classmethod
    def get_active(cls) -> Optional["PyFlameDebugger"]:
        """Get the currently active debugger instance."""
        return cls._instance

    def set_breakpoint(
        self,
        op: Optional[str] = None,
        tensor: Optional[str] = None,
        condition: Optional[Callable[..., bool]] = None,
        callback: Optional[Callable[..., None]] = None,
    ) -> int:
        """Set a breakpoint.

        Args:
            op: Operation name to break on (e.g., "matmul", "relu")
            tensor: Tensor name to watch
            condition: Custom condition function
            callback: Callback when breakpoint is hit

        Returns:
            Breakpoint ID

        Example:
            >>> bp_id = debugger.set_breakpoint(op="matmul")
            >>> bp_id = debugger.set_breakpoint(
            ...     condition=lambda ctx: ctx.get("shape", [0])[0] > 1000
            ... )
        """
        PyFlameDebugger._breakpoint_counter += 1
        bp_id = PyFlameDebugger._breakpoint_counter

        if op is not None:
            bp_type = BreakpointType.OPERATION
        elif tensor is not None:
            bp_type = BreakpointType.TENSOR_VALUE
        elif condition is not None:
            bp_type = BreakpointType.CUSTOM
        else:
            bp_type = BreakpointType.CUSTOM

        bp = Breakpoint(
            id=bp_id,
            type=bp_type,
            condition=condition,
            callback=callback,
            op_name=op,
            tensor_name=tensor,
        )

        self.breakpoints[bp_id] = bp

        if self.verbose:
            print(f"[Debugger] Breakpoint {bp_id} set: {bp_type.value}")

        return bp_id

    def remove_breakpoint(self, bp_id: int) -> bool:
        """Remove a breakpoint by ID.

        Args:
            bp_id: Breakpoint ID to remove

        Returns:
            True if removed, False if not found
        """
        if bp_id in self.breakpoints:
            del self.breakpoints[bp_id]
            if self.verbose:
                print(f"[Debugger] Breakpoint {bp_id} removed")
            return True
        return False

    def clear_breakpoints(self):
        """Remove all breakpoints."""
        self.breakpoints.clear()
        if self.verbose:
            print("[Debugger] All breakpoints cleared")

    def enable_breakpoint(self, bp_id: int, enabled: bool = True):
        """Enable or disable a breakpoint.

        Args:
            bp_id: Breakpoint ID
            enabled: Whether to enable
        """
        if bp_id in self.breakpoints:
            self.breakpoints[bp_id].enabled = enabled

    def watch(self, name: str, tensor):
        """Add a tensor to the watch list.

        Args:
            name: Name for the watched tensor
            tensor: Tensor to watch
        """
        self.watch_tensors[name] = tensor
        if self.verbose:
            print(f"[Debugger] Watching tensor: {name}")

    def unwatch(self, name: str):
        """Remove a tensor from the watch list.

        Args:
            name: Name of tensor to unwatch
        """
        if name in self.watch_tensors:
            del self.watch_tensors[name]

    def inspect(self, tensor) -> Dict[str, Any]:
        """Inspect a tensor.

        Args:
            tensor: Tensor to inspect

        Returns:
            Dictionary with tensor information
        """
        return self._inspector.inspect(tensor)

    def compare(self, tensor1, tensor2, **kwargs) -> Dict[str, Any]:
        """Compare two tensors.

        Args:
            tensor1: First tensor
            tensor2: Second tensor
            **kwargs: Additional arguments for comparison

        Returns:
            Comparison results
        """
        return self._inspector.compare(tensor1, tensor2, **kwargs)

    def print_watches(self):
        """Print information about all watched tensors."""
        print("\n=== Watched Tensors ===")
        for name, tensor in self.watch_tensors.items():
            info = self.inspect(tensor)
            print(f"\n{name}:")
            print(f"  Shape: {info['shape']}")
            print(f"  DType: {info['dtype']}")
            print(f"  Evaluated: {info['is_evaluated']}")
            if info.get("stats"):
                stats = info["stats"]
                print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
                print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        print("=" * 24)

    def print_breakpoints(self):
        """Print all breakpoints."""
        print("\n=== Breakpoints ===")
        for bp_id, bp in self.breakpoints.items():
            status = "enabled" if bp.enabled else "disabled"
            print(f"  [{bp_id}] {bp.type.value} ({status}) - hits: {bp.hit_count}")
            if bp.op_name:
                print(f"       op: {bp.op_name}")
            if bp.tensor_name:
                print(f"       tensor: {bp.tensor_name}")
        print("=" * 20)

    def _check_breakpoints(self, context: Dict[str, Any]) -> Optional[Breakpoint]:
        """Check if any breakpoint should trigger.

        Args:
            context: Current execution context

        Returns:
            Triggered breakpoint or None
        """
        for bp in self.breakpoints.values():
            if not bp.enabled:
                continue

            # Check operation breakpoints
            if bp.type == BreakpointType.OPERATION:
                if bp.op_name and context.get("op") == bp.op_name:
                    bp.hit_count += 1
                    return bp

            # Check custom condition breakpoints
            if bp.type == BreakpointType.CUSTOM:
                if bp.check(context):
                    bp.hit_count += 1
                    return bp

        return None

    def _on_breakpoint(self, bp: Breakpoint, context: Dict[str, Any]):
        """Handle a triggered breakpoint.

        Args:
            bp: Triggered breakpoint
            context: Execution context
        """
        print(f"\n[Debugger] Breakpoint {bp.id} hit ({bp.type.value})")

        if context.get("op"):
            print(f"  Operation: {context['op']}")
        if context.get("shape"):
            print(f"  Shape: {context['shape']}")

        if bp.callback:
            bp.callback(context)

        self.history.append(
            {
                "breakpoint_id": bp.id,
                "context": context.copy(),
            }
        )


# Global debugger functions
_global_breakpoints: Dict[int, Breakpoint] = {}


def set_breakpoint(
    op: Optional[str] = None,
    condition: Optional[Callable[..., bool]] = None,
    callback: Optional[Callable[..., None]] = None,
) -> int:
    """Set a global breakpoint.

    Args:
        op: Operation name to break on
        condition: Custom condition function
        callback: Callback when breakpoint is hit

    Returns:
        Breakpoint ID

    Example:
        >>> bp_id = pf.tools.set_breakpoint(op="matmul")
    """
    debugger = PyFlameDebugger.get_active()
    if debugger:
        return debugger.set_breakpoint(op=op, condition=condition, callback=callback)

    # Store globally if no active debugger
    PyFlameDebugger._breakpoint_counter += 1
    bp_id = PyFlameDebugger._breakpoint_counter

    bp_type = BreakpointType.OPERATION if op else BreakpointType.CUSTOM
    bp = Breakpoint(
        id=bp_id,
        type=bp_type,
        condition=condition,
        callback=callback,
        op_name=op,
    )
    _global_breakpoints[bp_id] = bp
    return bp_id


def clear_breakpoints():
    """Clear all global breakpoints."""
    _global_breakpoints.clear()
    debugger = PyFlameDebugger.get_active()
    if debugger:
        debugger.clear_breakpoints()


def debug_op(func: Callable) -> Callable:
    """Decorator to add debugging hooks to an operation.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with debug hooks

    Example:
        >>> @debug_op
        ... def my_operation(x, y):
        ...     return x + y
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        debugger = PyFlameDebugger.get_active()

        if debugger:
            context = {
                "op": func.__name__,
                "args": args,
                "kwargs": kwargs,
            }

            bp = debugger._check_breakpoints(context)
            if bp:
                debugger._on_breakpoint(bp, context)

        result = func(*args, **kwargs)

        if debugger:
            context["result"] = result
            debugger.history.append({"op": func.__name__, "context": context})

        return result

    return wrapper
