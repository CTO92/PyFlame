"""
PyFlame: Native deep learning framework for Cerebras WSE

PRE-RELEASE ALPHA 1.0
This software is in early development and is not yet ready for production use.
APIs may change without notice. Use at your own risk.

A tensor computation library designed natively for the Cerebras Wafer-Scale Engine,
featuring lazy evaluation, automatic CSL code generation, and Python-first API.
"""

# Track whether C++ bindings are available
_CPP_AVAILABLE = False

try:
    from ._pyflame_cpp import (  # CSL code generation; Data types; Layout; Tensor class; Graph access; Elementwise math; Tensor combination; Matrix operations; Activation functions
        CodeGenOptions,
        CodeGenResult,
        CSLCodeGenerator,
        DType,
        Graph,
        MeshLayout,
        Node,
        PECoord,
        Tensor,
        TensorSpec,
        abs,
        cat,
        compile_to_csl,
        cos,
        dtype_name,
        dtype_size,
        exp,
        gelu,
        get_graph,
        get_node,
        log,
        log_softmax,
        matmul,
        relu,
        sigmoid,
        silu,
        sin,
        softmax,
        sqrt,
        stack,
        tanh,
    )

    _CPP_AVAILABLE = True
except ImportError:
    # C++ bindings not built - provide placeholder types for ecosystem modules
    # Core tensor operations will not work, but tools/integrations can be imported
    import warnings

    warnings.warn(
        "PyFlame C++ bindings not found. Core tensor operations unavailable. "
        "Install from source with 'pip install -e .' to build C++ extensions.",
        ImportWarning,
    )

    # Placeholder types for when C++ is not available
    DType = None
    dtype_size = None
    dtype_name = None
    PECoord = None
    MeshLayout = None
    Tensor = None
    matmul = None
    relu = None
    sigmoid = None
    tanh = None
    gelu = None
    silu = None
    softmax = None
    log_softmax = None
    abs = None
    sqrt = None
    exp = None
    log = None
    sin = None
    cos = None
    cat = None
    stack = None
    TensorSpec = None
    Node = None
    Graph = None
    get_graph = None
    get_node = None
    CodeGenOptions = None
    CodeGenResult = None
    CSLCodeGenerator = None
    compile_to_csl = None

# Convenient dtype aliases
if _CPP_AVAILABLE:
    float32 = DType.float32
    float16 = DType.float16
    bfloat16 = DType.bfloat16
    int32 = DType.int32
    int16 = DType.int16
    int8 = DType.int8
    bool_ = DType.bool_
else:
    float32 = None
    float16 = None
    bfloat16 = None
    int32 = None
    int16 = None
    int8 = None
    bool_ = None


def _require_cpp(func_name: str):
    """Raise an error if C++ bindings are not available."""
    if not _CPP_AVAILABLE:
        raise RuntimeError(
            f"{func_name}() requires PyFlame C++ bindings. "
            "Build from source with 'pip install -e .' to enable core tensor operations."
        )


# Factory functions with nicer API
def zeros(shape, dtype=None, layout=None):
    """Create a tensor filled with zeros.

    Args:
        shape: Tuple or list of dimensions.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor filled with zeros.

    Example:
        >>> x = pf.zeros([3, 4])
        >>> x.shape
        [3, 4]
    """
    _require_cpp("zeros")
    if dtype is None:
        dtype = float32
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.zeros(list(shape), dtype, layout)


def ones(shape, dtype=None, layout=None):
    """Create a tensor filled with ones.

    Args:
        shape: Tuple or list of dimensions.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor filled with ones.
    """
    _require_cpp("ones")
    if dtype is None:
        dtype = float32
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.ones(list(shape), dtype, layout)


def full(shape, value, dtype=None, layout=None):
    """Create a tensor filled with a scalar value.

    Args:
        shape: Tuple or list of dimensions.
        value: Scalar value to fill.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor filled with the given value.
    """
    _require_cpp("full")
    if dtype is None:
        dtype = float32
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.full(list(shape), float(value), dtype, layout)


def randn(shape, dtype=None, layout=None):
    """Create a tensor with random normal values (mean=0, std=1).

    Args:
        shape: Tuple or list of dimensions.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor with random normal values.
    """
    _require_cpp("randn")
    if dtype is None:
        dtype = float32
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.randn(list(shape), dtype, layout)


def rand(shape, dtype=None, layout=None):
    """Create a tensor with random uniform values in [0, 1).

    Args:
        shape: Tuple or list of dimensions.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor with random uniform values.
    """
    _require_cpp("rand")
    if dtype is None:
        dtype = float32
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.rand(list(shape), dtype, layout)


def arange(start, end=None, step=1, dtype=None):
    """Create a 1D tensor with values from start to end.

    Args:
        start: Start value (or end if end is None).
        end: End value (exclusive).
        step: Step size (default: 1). Must be non-zero.
        dtype: Data type (default: float32).

    Returns:
        A 1D Tensor with evenly spaced values.

    Example:
        >>> pf.arange(5)          # [0, 1, 2, 3, 4]
        >>> pf.arange(1, 5)       # [1, 2, 3, 4]
        >>> pf.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
    """
    _require_cpp("arange")
    if dtype is None:
        dtype = float32
    if end is None:
        end = start
        start = 0
    if step == 0:
        raise ValueError("step must be non-zero")
    # Use float conversion to preserve precision, then convert based on dtype
    # For integer dtypes, we still need integer steps
    return Tensor.arange(float(start), float(end), float(step), dtype)


def tensor(data, dtype=None):
    """Create a tensor from Python data or numpy array.

    Args:
        data: Python list, numpy array, or scalar.
        dtype: Data type (default: float32).

    Returns:
        A new Tensor containing the data.

    Example:
        >>> pf.tensor([[1, 2], [3, 4]])
        >>> pf.tensor(np.random.randn(3, 4))
    """
    _require_cpp("tensor")
    import numpy as np

    # Map PyFlame dtype to numpy dtype
    if dtype is None:
        np_dtype = np.float32
    elif dtype == float32:
        np_dtype = np.float32
    elif dtype == float16:
        np_dtype = np.float16
    elif dtype == int32:
        np_dtype = np.int32
    elif dtype == int16:
        np_dtype = np.int16
    elif dtype == int8:
        np_dtype = np.int8
    elif dtype == bool_:
        np_dtype = np.bool_
    else:
        np_dtype = np.float32  # Default fallback

    arr = np.asarray(data, dtype=np_dtype)
    return Tensor.from_numpy(arr)


def from_numpy(arr, dtype=None):
    """Create a tensor from a numpy array.

    Args:
        arr: NumPy array.
        dtype: Optional PyFlame dtype. If None, infers from numpy array dtype.
            Supported numpy dtypes: float32, float16, int32, int16, int8, bool.
            Other dtypes will be converted to float32 with a warning.

    Returns:
        A new Tensor containing the data.
    """
    _require_cpp("from_numpy")
    import warnings

    import numpy as np

    arr = np.ascontiguousarray(arr)

    # Map numpy dtypes to supported types
    supported_dtypes = {
        np.float32: np.float32,
        np.float16: np.float16,
        np.int32: np.int32,
        np.int16: np.int16,
        np.int8: np.int8,
        np.bool_: np.bool_,
    }

    # If dtype is explicitly provided, convert to it
    if dtype is not None:
        # Map PyFlame dtype to numpy dtype
        dtype_map = {
            float32: np.float32,
            float16: np.float16,
            bfloat16: np.float32,  # bfloat16 not natively supported in numpy
            int32: np.int32,
            int16: np.int16,
            int8: np.int8,
            bool_: np.bool_,
        }
        target_dtype = dtype_map.get(dtype, np.float32)
        arr = arr.astype(target_dtype)
    elif arr.dtype.type not in supported_dtypes:
        # Warn about automatic conversion
        warnings.warn(
            f"NumPy dtype {arr.dtype} is not directly supported. "
            f"Converting to float32. Use dtype parameter to specify target type.",
            UserWarning,
            stacklevel=2,
        )
        arr = arr.astype(np.float32)

    return Tensor.from_numpy(arr)


def is_lazy(t):
    """Check if a tensor has not been evaluated yet.

    Args:
        t: Tensor to check.

    Returns:
        True if the tensor is lazy (not yet computed).
    """
    _require_cpp("is_lazy")
    return not t.is_evaluated()


def eval(*tensors):
    """Force evaluation of one or more tensors.

    Args:
        *tensors: Tensors to evaluate.

    Returns:
        The evaluated tensor(s), or None if no tensors provided.

    Example:
        >>> a = pf.randn([100])
        >>> b = pf.randn([100])
        >>> c = a + b  # Lazy
        >>> pf.eval(c) # Now computed
    """
    _require_cpp("eval")
    if len(tensors) == 0:
        return None
    for t in tensors:
        t.eval()
    return tensors[0] if len(tensors) == 1 else tensors


def print_graph(t):
    """Print the computation graph for a tensor.

    Args:
        t: Tensor whose graph to print.
    """
    _require_cpp("print_graph")
    graph = get_graph(t)
    if graph:
        print(repr(graph))
    else:
        print("No graph associated with tensor")


# Version (Pre-Release Alpha 1.0)
try:
    from ._version import __version__
except ImportError:
    __version__ = "1.0.0-alpha"

__version_info__ = (1, 0, 0, "alpha")
__release_status__ = "Pre-Release Alpha"

__all__ = [
    # Types
    "DType",
    "PECoord",
    "MeshLayout",
    "Tensor",
    "TensorSpec",
    "Node",
    "Graph",
    # Dtype aliases
    "float32",
    "float16",
    "bfloat16",
    "int32",
    "int16",
    "int8",
    "bool_",
    # Factory functions
    "zeros",
    "ones",
    "full",
    "randn",
    "rand",
    "arange",
    "tensor",
    "from_numpy",
    # Operations
    "matmul",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "abs",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "cat",
    "stack",
    # Utilities
    "is_lazy",
    "eval",
    "print_graph",
    "get_graph",
    "get_node",
    "dtype_size",
    "dtype_name",
    # CSL
    "CodeGenOptions",
    "CodeGenResult",
    "CSLCodeGenerator",
    "compile_to_csl",
    # Version and availability
    "__version__",
    "__version_info__",
    "__release_status__",
    "_CPP_AVAILABLE",
]
