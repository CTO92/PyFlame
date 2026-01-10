"""
PyFlame: Native deep learning framework for Cerebras WSE

PRE-RELEASE ALPHA 1.0
This software is in early development and is not yet ready for production use.
APIs may change without notice. Use at your own risk.

A tensor computation library designed natively for the Cerebras Wafer-Scale Engine,
featuring lazy evaluation, automatic CSL code generation, and Python-first API.
"""

from ._pyflame_cpp import (
    # Data types
    DType,
    dtype_size,
    dtype_name,

    # Layout
    PECoord,
    MeshLayout,

    # Tensor class
    Tensor,

    # Matrix operations
    matmul,

    # Activation functions
    relu,
    sigmoid,
    tanh,
    gelu,
    silu,
    softmax,
    log_softmax,

    # Elementwise math
    abs,
    sqrt,
    exp,
    log,
    sin,
    cos,

    # Tensor combination
    cat,
    stack,

    # Graph access
    TensorSpec,
    Node,
    Graph,
    get_graph,
    get_node,

    # CSL code generation
    CodeGenOptions,
    CodeGenResult,
    CSLCodeGenerator,
    compile_to_csl,
)

# Convenient dtype aliases
float32 = DType.float32
float16 = DType.float16
bfloat16 = DType.bfloat16
int32 = DType.int32
int16 = DType.int16
int8 = DType.int8
bool_ = DType.bool_


# Factory functions with nicer API
def zeros(shape, dtype=float32, layout=None):
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
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.zeros(list(shape), dtype, layout)


def ones(shape, dtype=float32, layout=None):
    """Create a tensor filled with ones.

    Args:
        shape: Tuple or list of dimensions.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor filled with ones.
    """
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.ones(list(shape), dtype, layout)


def full(shape, value, dtype=float32, layout=None):
    """Create a tensor filled with a scalar value.

    Args:
        shape: Tuple or list of dimensions.
        value: Scalar value to fill.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor filled with the given value.
    """
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.full(list(shape), float(value), dtype, layout)


def randn(shape, dtype=float32, layout=None):
    """Create a tensor with random normal values (mean=0, std=1).

    Args:
        shape: Tuple or list of dimensions.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor with random normal values.
    """
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.randn(list(shape), dtype, layout)


def rand(shape, dtype=float32, layout=None):
    """Create a tensor with random uniform values in [0, 1).

    Args:
        shape: Tuple or list of dimensions.
        dtype: Data type (default: float32).
        layout: MeshLayout for PE distribution (default: SinglePE).

    Returns:
        A new Tensor with random uniform values.
    """
    if layout is None:
        layout = MeshLayout.single_pe()
    return Tensor.rand(list(shape), dtype, layout)


def arange(start, end=None, step=1, dtype=float32):
    """Create a 1D tensor with values from start to end.

    Args:
        start: Start value (or end if end is None).
        end: End value (exclusive).
        step: Step size (default: 1).
        dtype: Data type (default: float32).

    Returns:
        A 1D Tensor with evenly spaced values.

    Example:
        >>> pf.arange(5)          # [0, 1, 2, 3, 4]
        >>> pf.arange(1, 5)       # [1, 2, 3, 4]
        >>> pf.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
    """
    if end is None:
        end = start
        start = 0
    return Tensor.arange(int(start), int(end), int(step), dtype)


def tensor(data, dtype=None):
    """Create a tensor from Python data or numpy array.

    Args:
        data: Python list, numpy array, or scalar.
        dtype: Data type (default: inferred from data).

    Returns:
        A new Tensor containing the data.

    Example:
        >>> pf.tensor([[1, 2], [3, 4]])
        >>> pf.tensor(np.random.randn(3, 4))
    """
    import numpy as np
    arr = np.asarray(data, dtype=np.float32)
    return Tensor.from_numpy(arr)


def from_numpy(arr):
    """Create a tensor from a numpy array.

    Args:
        arr: NumPy array.

    Returns:
        A new Tensor containing the data.
    """
    import numpy as np
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return Tensor.from_numpy(arr)


def is_lazy(t):
    """Check if a tensor has not been evaluated yet.

    Args:
        t: Tensor to check.

    Returns:
        True if the tensor is lazy (not yet computed).
    """
    return not t.is_evaluated()


def eval(*tensors):
    """Force evaluation of one or more tensors.

    Args:
        *tensors: Tensors to evaluate.

    Returns:
        The evaluated tensor(s).

    Example:
        >>> a = pf.randn([100])
        >>> b = pf.randn([100])
        >>> c = a + b  # Lazy
        >>> pf.eval(c) # Now computed
    """
    for t in tensors:
        t.eval()
    return tensors[0] if len(tensors) == 1 else tensors


def print_graph(t):
    """Print the computation graph for a tensor.

    Args:
        t: Tensor whose graph to print.
    """
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
    'DType', 'PECoord', 'MeshLayout', 'Tensor',
    'TensorSpec', 'Node', 'Graph',

    # Dtype aliases
    'float32', 'float16', 'bfloat16', 'int32', 'int16', 'int8', 'bool_',

    # Factory functions
    'zeros', 'ones', 'full', 'randn', 'rand', 'arange', 'tensor', 'from_numpy',

    # Operations
    'matmul', 'relu', 'sigmoid', 'tanh', 'gelu', 'silu', 'softmax', 'log_softmax',
    'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'cat', 'stack',

    # Utilities
    'is_lazy', 'eval', 'print_graph', 'get_graph', 'get_node',
    'dtype_size', 'dtype_name',

    # CSL
    'CodeGenOptions', 'CodeGenResult', 'CSLCodeGenerator', 'compile_to_csl',

    # Version
    '__version__',
    '__version_info__',
    '__release_status__',
]
