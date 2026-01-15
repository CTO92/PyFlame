"""
Custom operator registration for PyFlame.

Allows users to define and register custom operations.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class CustomOp:
    """Custom operator definition.

    Attributes:
        name: Unique operator name
        forward_fn: Forward computation function
        backward_fn: Backward gradient function (optional)
        csl_template: CSL code template for Cerebras execution (optional)
        schema: Operator schema definition
        doc: Documentation string
    """

    name: str
    forward_fn: Callable
    backward_fn: Optional[Callable] = None
    csl_template: Optional[str] = None
    schema: Optional[str] = None
    doc: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        """Execute the custom operator."""
        return self.forward_fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"CustomOp(name={self.name!r})"


# Global registry of custom operators
_custom_ops: Dict[str, CustomOp] = {}


def register_custom_op(
    name: str,
    forward_fn: Callable,
    backward_fn: Optional[Callable] = None,
    csl_template: Optional[str] = None,
    schema: Optional[str] = None,
    doc: str = "",
    **metadata,
) -> CustomOp:
    """Register a custom operator.

    Creates and registers a new operator that can be used in PyFlame
    computation graphs.

    Args:
        name: Unique operator name
        forward_fn: Forward computation function
        backward_fn: Backward gradient function for autograd
        csl_template: CSL code template for Cerebras execution
        schema: Operator schema (e.g., "(Tensor x, Tensor y) -> Tensor")
        doc: Documentation string
        **metadata: Additional metadata

    Returns:
        Registered CustomOp instance

    Example:
        >>> def swish_forward(x):
        ...     return x * pf.sigmoid(x)
        ...
        >>> def swish_backward(grad_output, x):
        ...     sig = pf.sigmoid(x)
        ...     return grad_output * (sig + x * sig * (1 - sig))
        ...
        >>> swish = register_custom_op(
        ...     "swish",
        ...     forward_fn=swish_forward,
        ...     backward_fn=swish_backward,
        ...     doc="Swish/SiLU activation function"
        ... )
        >>> y = swish(x)
    """
    if name in _custom_ops:
        raise ValueError(f"Custom op '{name}' already registered")

    op = CustomOp(
        name=name,
        forward_fn=forward_fn,
        backward_fn=backward_fn,
        csl_template=csl_template,
        schema=schema,
        doc=doc,
        metadata=metadata,
    )

    _custom_ops[name] = op

    # Try to register with C++ backend if available
    try:
        import pyflame as pf

        if hasattr(pf, "_register_custom_op"):
            pf._register_custom_op(name, forward_fn, backward_fn, csl_template)
    except (ImportError, AttributeError):
        pass

    return op


def custom_op(
    name: str,
    backward_fn: Optional[Callable] = None,
    csl_template: Optional[str] = None,
    schema: Optional[str] = None,
    **metadata,
):
    """Decorator for registering custom operators.

    Convenient way to register a function as a custom operator.

    Args:
        name: Unique operator name
        backward_fn: Backward gradient function
        csl_template: CSL code template
        schema: Operator schema
        **metadata: Additional metadata

    Returns:
        Decorator function

    Example:
        >>> @custom_op("my_relu6")
        ... def relu6(x):
        ...     return pf.clamp(pf.relu(x), max=6.0)
        ...
        >>> y = relu6(x)

        >>> @custom_op("my_gelu", schema="(Tensor x) -> Tensor")
        ... def my_gelu(x):
        ...     return x * pf.sigmoid(1.702 * x)
    """

    def decorator(fn: Callable) -> CustomOp:
        return register_custom_op(
            name=name,
            forward_fn=fn,
            backward_fn=backward_fn,
            csl_template=csl_template,
            schema=schema,
            doc=fn.__doc__ or "",
            **metadata,
        )

    return decorator


def get_custom_op(name: str) -> Optional[CustomOp]:
    """Get a registered custom operator by name.

    Args:
        name: Operator name

    Returns:
        CustomOp instance or None if not found
    """
    return _custom_ops.get(name)


def list_custom_ops() -> List[str]:
    """List all registered custom operators.

    Returns:
        List of operator names
    """
    return list(_custom_ops.keys())


def unregister_custom_op(name: str) -> bool:
    """Unregister a custom operator.

    Args:
        name: Operator name

    Returns:
        True if removed, False if not found
    """
    if name in _custom_ops:
        del _custom_ops[name]
        return True
    return False


def clear_custom_ops():
    """Clear all registered custom operators."""
    _custom_ops.clear()


class AutogradFunction:
    """Base class for autograd-compatible custom operations.

    Provides a more structured way to define custom operations with
    automatic differentiation support.

    Example:
        >>> class MySoftplus(AutogradFunction):
        ...     @staticmethod
        ...     def forward(ctx, x, beta=1.0):
        ...         ctx.save_for_backward(x)
        ...         ctx.beta = beta
        ...         return (1 / beta) * pf.log(1 + pf.exp(beta * x))
        ...
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         x, = ctx.saved_tensors
        ...         beta = ctx.beta
        ...         return grad_output * pf.sigmoid(beta * x)
        ...
        >>> softplus = MySoftplus.apply
        >>> y = softplus(x, beta=2.0)
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Forward pass.

        Args:
            ctx: Context object for saving tensors
            *args: Input arguments
            **kwargs: Keyword arguments

        Returns:
            Operation output
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass.

        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient of the output

        Returns:
            Gradients with respect to inputs
        """
        raise NotImplementedError("Subclasses must implement backward()")

    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply the function.

        This is the main entry point for using the function.
        """
        ctx = _FunctionContext()
        output = cls.forward(ctx, *args, **kwargs)

        # Store context for backward pass
        if hasattr(output, "_autograd_ctx"):
            output._autograd_ctx = (cls, ctx)

        return output


class _FunctionContext:
    """Context object for storing tensors during forward pass."""

    def __init__(self):
        self.saved_tensors: Tuple = ()
        self._saved_data: Dict[str, Any] = {}

    def save_for_backward(self, *tensors):
        """Save tensors for the backward pass.

        Args:
            *tensors: Tensors to save
        """
        self.saved_tensors = tensors

    def __setattr__(self, name: str, value: Any):
        if name in ("saved_tensors", "_saved_data"):
            super().__setattr__(name, value)
        else:
            self._saved_data[name] = value

    def __getattr__(self, name: str) -> Any:
        if name == "_saved_data":
            return super().__getattribute__("_saved_data")
        return self._saved_data.get(name)


# Common custom operations
def _register_builtin_ops():
    """Register common custom operations."""

    # Swish/SiLU activation
    @custom_op("swish", schema="(Tensor x) -> Tensor")
    def swish(x):
        """Swish/SiLU activation: x * sigmoid(x)"""
        try:
            import pyflame as pf

            return x * pf.sigmoid(x)
        except ImportError:
            import numpy as np

            return x * (1 / (1 + np.exp(-x)))

    # Mish activation
    @custom_op("mish", schema="(Tensor x) -> Tensor")
    def mish(x):
        """Mish activation: x * tanh(softplus(x))"""
        try:
            import pyflame as pf

            return x * pf.tanh(pf.log(1 + pf.exp(x)))
        except ImportError:
            import numpy as np

            return x * np.tanh(np.log(1 + np.exp(x)))

    # Hard Swish
    @custom_op("hard_swish", schema="(Tensor x) -> Tensor")
    def hard_swish(x):
        """Hard Swish activation: x * relu6(x + 3) / 6"""
        try:
            import pyflame as pf

            return x * pf.clamp(x + 3, min=0, max=6) / 6
        except ImportError:
            import numpy as np

            return x * np.clip(x + 3, 0, 6) / 6

    # GELU approximation
    @custom_op("gelu_approx", schema="(Tensor x) -> Tensor")
    def gelu_approx(x):
        """Fast GELU approximation: x * sigmoid(1.702 * x)"""
        try:
            import pyflame as pf

            return x * pf.sigmoid(1.702 * x)
        except ImportError:
            import numpy as np

            return x * (1 / (1 + np.exp(-1.702 * x)))


# Register built-in ops on module load
try:
    _register_builtin_ops()
except Exception:
    pass  # Ignore errors during import
