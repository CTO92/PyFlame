# PyFlame API Reference

**PyFlame Version:** Pre-Release Alpha 1.0

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs described here are subject to change.

---

## Table of Contents

1. [Data Types](#data-types)
2. [Tensor Class](#tensor-class)
3. [Tensor Creation Functions](#tensor-creation-functions)
4. [Arithmetic Operations](#arithmetic-operations)
5. [Matrix Operations](#matrix-operations)
6. [Activation Functions](#activation-functions)
7. [Math Functions](#math-functions)
8. [Reduction Operations](#reduction-operations)
9. [Shape Operations](#shape-operations)
10. [Tensor Combination](#tensor-combination)
11. [Layouts](#layouts)
12. [Graph Inspection](#graph-inspection)
13. [CSL Code Generation](#csl-code-generation)
14. [Utility Functions](#utility-functions)

---

## Data Types

### DType Enum

PyFlame supports the following data types:

| Type | Python | C++ | Size | Description |
|------|--------|-----|------|-------------|
| `float32` | `pf.float32` | `DType::float32` | 4 bytes | Single precision float (default) |
| `float16` | `pf.float16` | `DType::float16` | 2 bytes | Half precision float |
| `bfloat16` | `pf.bfloat16` | `DType::bfloat16` | 2 bytes | Brain floating point |
| `int32` | `pf.int32` | `DType::int32` | 4 bytes | 32-bit signed integer |
| `int16` | `pf.int16` | `DType::int16` | 2 bytes | 16-bit signed integer |
| `int8` | `pf.int8` | `DType::int8` | 1 byte | 8-bit signed integer |
| `bool_` | `pf.bool_` | `DType::bool_` | 1 byte | Boolean |

### Type Functions

#### `dtype_size(dtype)`

Returns the size in bytes of a data type.

```python
>>> pf.dtype_size(pf.float32)
4
>>> pf.dtype_size(pf.float16)
2
```

#### `dtype_name(dtype)`

Returns the string name of a data type.

```python
>>> pf.dtype_name(pf.float32)
'float32'
```

---

## Tensor Class

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `list[int]` | Tensor dimensions |
| `dtype` | `DType` | Data type |
| `ndim` | `int` | Number of dimensions |
| `numel` | `int` | Total number of elements |
| `layout` | `MeshLayout` | PE distribution layout |

```python
>>> x = pf.randn([3, 4, 5])
>>> x.shape
[3, 4, 5]
>>> x.dtype
DType.float32
>>> x.ndim
3
>>> x.numel
60
```

### Methods

#### `eval()`

Force evaluation of the tensor's computation graph.

```python
x = pf.randn([100, 100])
y = pf.relu(x)
y.eval()  # Computation happens here
```

#### `is_evaluated()`

Check if tensor has been evaluated.

```python
>>> x = pf.randn([10])
>>> x.is_evaluated()
False
>>> x.eval()
>>> x.is_evaluated()
True
```

#### `numpy()`

Convert tensor to NumPy array. Triggers evaluation if needed.

```python
>>> x = pf.randn([3, 4])
>>> arr = x.numpy()
>>> type(arr)
<class 'numpy.ndarray'>
```

#### `sum(dim=None, keepdim=False)`

Sum elements along dimension(s).

```python
>>> x = pf.randn([3, 4])
>>> x.sum()          # Sum all elements -> scalar
>>> x.sum(dim=0)     # Sum along rows -> [4]
>>> x.sum(dim=1)     # Sum along columns -> [3]
>>> x.sum(dim=1, keepdim=True)  # Keep dims -> [3, 1]
```

#### `mean(dim=None, keepdim=False)`

Compute mean along dimension(s).

```python
>>> x = pf.randn([3, 4])
>>> x.mean()         # Mean of all elements
>>> x.mean(dim=0)    # Mean along rows
```

#### `max(dim=None, keepdim=False)`

Find maximum along dimension(s).

```python
>>> x = pf.randn([3, 4])
>>> x.max()          # Global maximum
>>> x.max(dim=1)     # Maximum per row
```

#### `min(dim=None, keepdim=False)`

Find minimum along dimension(s).

```python
>>> x = pf.randn([3, 4])
>>> x.min()          # Global minimum
>>> x.min(dim=0)     # Minimum per column
```

#### `reshape(new_shape)`

Reshape tensor to new dimensions.

```python
>>> x = pf.randn([12])
>>> y = x.reshape([3, 4])
>>> y.shape
[3, 4]
```

#### `transpose(dim0, dim1)`

Swap two dimensions.

```python
>>> x = pf.randn([3, 4])
>>> y = x.transpose(0, 1)
>>> y.shape
[4, 3]
```

#### `squeeze(dim=None)`

Remove dimensions of size 1.

```python
>>> x = pf.randn([1, 3, 1, 4])
>>> x.squeeze().shape      # [3, 4]
>>> x.squeeze(0).shape     # [3, 1, 4]
```

#### `unsqueeze(dim)`

Add a dimension of size 1.

```python
>>> x = pf.randn([3, 4])
>>> x.unsqueeze(0).shape   # [1, 3, 4]
>>> x.unsqueeze(2).shape   # [3, 4, 1]
```

---

## Tensor Creation Functions

### `zeros(shape, dtype=float32, layout=None)`

Create tensor filled with zeros.

**Parameters:**
- `shape`: List or tuple of dimensions
- `dtype`: Data type (default: `float32`)
- `layout`: MeshLayout for PE distribution (default: `single_pe()`)

```python
>>> pf.zeros([3, 4])
>>> pf.zeros([100, 100], dtype=pf.float16)
>>> pf.zeros([1024, 1024], layout=pf.MeshLayout.grid(4, 4))
```

### `ones(shape, dtype=float32, layout=None)`

Create tensor filled with ones.

```python
>>> pf.ones([3, 4])
>>> pf.ones([100], dtype=pf.int32)
```

### `full(shape, value, dtype=float32, layout=None)`

Create tensor filled with a specific value.

**Parameters:**
- `shape`: Dimensions
- `value`: Scalar fill value
- `dtype`: Data type
- `layout`: MeshLayout

```python
>>> pf.full([3, 4], 3.14)
>>> pf.full([10], -1, dtype=pf.int32)
```

### `randn(shape, dtype=float32, layout=None)`

Create tensor with random values from standard normal distribution (mean=0, std=1).

```python
>>> x = pf.randn([1000])
>>> pf.eval(x)
>>> x.numpy().mean()  # Approximately 0
>>> x.numpy().std()   # Approximately 1
```

### `rand(shape, dtype=float32, layout=None)`

Create tensor with random values uniformly distributed in [0, 1).

```python
>>> x = pf.rand([1000])
>>> pf.eval(x)
>>> x.numpy().min()  # >= 0
>>> x.numpy().max()  # < 1
```

### `arange(start, end=None, step=1, dtype=float32)`

Create 1D tensor with evenly spaced values.

**Parameters:**
- `start`: Start value (or end if `end` is None)
- `end`: End value (exclusive)
- `step`: Step size

```python
>>> pf.arange(5)           # [0, 1, 2, 3, 4]
>>> pf.arange(1, 5)        # [1, 2, 3, 4]
>>> pf.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
>>> pf.arange(0, 5, dtype=pf.int32)
```

### `tensor(data, dtype=None)`

Create tensor from Python list or NumPy array.

```python
>>> pf.tensor([[1, 2, 3], [4, 5, 6]])
>>> pf.tensor(np.random.randn(3, 4))
```

### `from_numpy(arr)`

Create tensor from NumPy array.

```python
>>> arr = np.random.randn(3, 4).astype(np.float32)
>>> x = pf.from_numpy(arr)
```

---

## Arithmetic Operations

Tensors support standard arithmetic operators:

### Addition

```python
c = a + b       # Element-wise addition
c = a + 5.0     # Scalar addition
```

### Subtraction

```python
c = a - b       # Element-wise subtraction
c = a - 5.0     # Scalar subtraction
```

### Multiplication

```python
c = a * b       # Element-wise multiplication
c = a * 2.0     # Scalar multiplication
```

### Division

```python
c = a / b       # Element-wise division
c = a / 2.0     # Scalar division
```

### Negation

```python
c = -a          # Negate all elements
```

### Broadcasting

Operations follow NumPy broadcasting rules:

```python
>>> a = pf.randn([3, 4])
>>> b = pf.randn([4])      # Broadcasts to [3, 4]
>>> c = a + b              # Shape: [3, 4]

>>> a = pf.randn([3, 1])
>>> b = pf.randn([1, 4])
>>> c = a * b              # Shape: [3, 4]
```

---

## Matrix Operations

### `matmul(a, b)` / `a @ b`

Matrix multiplication.

**Parameters:**
- `a`: Left tensor [..., M, K]
- `b`: Right tensor [..., K, N]

**Returns:** Tensor [..., M, N]

```python
>>> a = pf.randn([100, 50])
>>> b = pf.randn([50, 75])
>>> c = a @ b           # Using operator
>>> c = pf.matmul(a, b) # Using function
>>> c.shape
[100, 75]
```

**Batch Matrix Multiplication:**

```python
>>> a = pf.randn([8, 100, 50])   # 8 matrices
>>> b = pf.randn([8, 50, 75])
>>> c = a @ b
>>> c.shape
[8, 100, 75]
```

---

## Activation Functions

### `relu(x)`

Rectified Linear Unit: max(0, x)

```python
>>> x = pf.tensor([-1, 0, 1, 2])
>>> pf.relu(x).numpy()
array([0., 0., 1., 2.])
```

### `sigmoid(x)`

Sigmoid: 1 / (1 + exp(-x))

```python
>>> x = pf.tensor([0])
>>> pf.sigmoid(x).numpy()
array([0.5])
```

### `tanh(x)`

Hyperbolic tangent.

```python
>>> x = pf.tensor([0])
>>> pf.tanh(x).numpy()
array([0.])
```

### `gelu(x)`

Gaussian Error Linear Unit.

```python
>>> x = pf.randn([100])
>>> y = pf.gelu(x)
```

### `silu(x)`

Sigmoid Linear Unit (Swish): x * sigmoid(x)

```python
>>> x = pf.randn([100])
>>> y = pf.silu(x)
```

### `softmax(x, dim=-1)`

Softmax along specified dimension.

**Parameters:**
- `x`: Input tensor
- `dim`: Dimension to normalize (default: -1, last dimension)

```python
>>> x = pf.randn([32, 10])  # 32 samples, 10 classes
>>> probs = pf.softmax(x, dim=1)
>>> probs.sum(dim=1)  # Each row sums to 1
```

### `log_softmax(x, dim=-1)`

Log softmax (numerically stable log(softmax(x))).

```python
>>> x = pf.randn([32, 10])
>>> log_probs = pf.log_softmax(x, dim=1)
```

---

## Math Functions

### `abs(x)`

Absolute value.

```python
>>> x = pf.tensor([-1, -2, 3])
>>> pf.abs(x).numpy()
array([1., 2., 3.])
```

### `sqrt(x)`

Square root (for non-negative values).

```python
>>> x = pf.tensor([1, 4, 9])
>>> pf.sqrt(x).numpy()
array([1., 2., 3.])
```

### `exp(x)`

Exponential.

```python
>>> x = pf.tensor([0, 1, 2])
>>> pf.exp(x).numpy()
array([1., 2.718..., 7.389...])
```

### `log(x)`

Natural logarithm (for positive values).

```python
>>> x = pf.tensor([1, 2.718, 7.389])
>>> pf.log(x).numpy()
array([0., 1., 2.])  # Approximately
```

### `sin(x)`

Sine.

```python
>>> x = pf.tensor([0, 3.14159/2])
>>> pf.sin(x).numpy()
array([0., 1.])  # Approximately
```

### `cos(x)`

Cosine.

```python
>>> x = pf.tensor([0, 3.14159])
>>> pf.cos(x).numpy()
array([1., -1.])  # Approximately
```

---

## Reduction Operations

All reduction operations support:
- `dim`: Dimension(s) to reduce (None = all)
- `keepdim`: Keep reduced dimensions as size 1

### `sum(x, dim=None, keepdim=False)`

Sum elements.

```python
>>> x = pf.tensor([[1, 2], [3, 4]])
>>> x.sum()           # 10
>>> x.sum(dim=0)      # [4, 6]
>>> x.sum(dim=1)      # [3, 7]
```

### `mean(x, dim=None, keepdim=False)`

Compute mean.

```python
>>> x = pf.tensor([[1, 2], [3, 4]])
>>> x.mean()          # 2.5
>>> x.mean(dim=0)     # [2, 3]
```

### `max(x, dim=None, keepdim=False)`

Find maximum.

```python
>>> x = pf.tensor([[1, 5], [3, 2]])
>>> x.max()           # 5
>>> x.max(dim=1)      # [5, 3]
```

### `min(x, dim=None, keepdim=False)`

Find minimum.

```python
>>> x = pf.tensor([[1, 5], [3, 2]])
>>> x.min()           # 1
>>> x.min(dim=0)      # [1, 2]
```

---

## Shape Operations

### `reshape(x, new_shape)`

Reshape tensor.

```python
>>> x = pf.arange(12)
>>> x.reshape([3, 4]).shape
[3, 4]
>>> x.reshape([2, 2, 3]).shape
[2, 2, 3]
```

Use -1 for automatic dimension:

```python
>>> x = pf.randn([24])
>>> x.reshape([4, -1]).shape  # [4, 6]
```

### `transpose(x, dim0, dim1)`

Swap two dimensions.

```python
>>> x = pf.randn([3, 4, 5])
>>> x.transpose(0, 2).shape
[5, 4, 3]
```

### `squeeze(x, dim=None)`

Remove size-1 dimensions.

```python
>>> x = pf.randn([1, 3, 1, 4, 1])
>>> x.squeeze().shape      # [3, 4]
>>> x.squeeze(0).shape     # [3, 1, 4, 1]
```

### `unsqueeze(x, dim)`

Add size-1 dimension.

```python
>>> x = pf.randn([3, 4])
>>> x.unsqueeze(0).shape   # [1, 3, 4]
>>> x.unsqueeze(-1).shape  # [3, 4, 1]
```

---

## Tensor Combination

### `cat(tensors, dim=0)`

Concatenate tensors along a dimension.

**Parameters:**
- `tensors`: List of tensors
- `dim`: Dimension to concatenate along

```python
>>> a = pf.randn([3, 4])
>>> b = pf.randn([5, 4])
>>> c = pf.cat([a, b], dim=0)
>>> c.shape
[8, 4]
```

### `stack(tensors, dim=0)`

Stack tensors along a new dimension.

**Parameters:**
- `tensors`: List of tensors (must have same shape)
- `dim`: Where to insert new dimension

```python
>>> a = pf.randn([3, 4])
>>> b = pf.randn([3, 4])
>>> c = pf.stack([a, b], dim=0)
>>> c.shape
[2, 3, 4]
```

---

## Layouts

### MeshLayout Class

Controls how tensors are distributed across Processing Elements (PEs) on the Cerebras WSE.

#### `MeshLayout.single_pe()`

All data on a single PE. Default for most tensors.

```python
>>> layout = pf.MeshLayout.single_pe()
>>> x = pf.zeros([100, 100], layout=layout)
```

#### `MeshLayout.row_partition(n)`

Distribute rows across n PEs.

```python
>>> layout = pf.MeshLayout.row_partition(4)
>>> x = pf.zeros([100, 100], layout=layout)
# Each PE holds 25 rows
```

#### `MeshLayout.col_partition(n)`

Distribute columns across n PEs.

```python
>>> layout = pf.MeshLayout.col_partition(4)
>>> x = pf.zeros([100, 100], layout=layout)
# Each PE holds 25 columns
```

#### `MeshLayout.grid(rows, cols)`

2D tiling across rows × cols PEs.

```python
>>> layout = pf.MeshLayout.grid(4, 4)
>>> x = pf.zeros([1024, 1024], layout=layout)
# Each PE holds 256x256 tile
```

### PECoord Class

Represents a Processing Element coordinate.

```python
>>> coord = pf.PECoord(2, 3)  # PE at row 2, column 3
>>> coord.row
2
>>> coord.col
3
```

---

## Graph Inspection

### `get_graph(tensor)`

Get the computation graph for a tensor.

```python
>>> a = pf.randn([100])
>>> b = pf.relu(a)
>>> graph = pf.get_graph(b)
>>> print(graph)
```

### `get_node(tensor)`

Get the IR node for a tensor.

```python
>>> a = pf.randn([100])
>>> node = pf.get_node(a)
```

### `print_graph(tensor)`

Print human-readable computation graph.

```python
>>> a = pf.randn([100, 100])
>>> b = pf.randn([100, 100])
>>> c = a @ b
>>> d = pf.relu(c)
>>> pf.print_graph(d)
```

### Graph Class

Represents the computation graph IR.

**Properties:**
- `nodes`: List of nodes in the graph
- `num_nodes`: Number of nodes

### Node Class

Represents a single operation in the graph.

**Properties:**
- `op_type`: Type of operation
- `inputs`: List of input nodes
- `output_shape`: Shape of output tensor
- `output_dtype`: Data type of output

### TensorSpec Class

Specification for a tensor (shape, dtype, layout).

---

## CSL Code Generation

### CodeGenOptions Class

Configuration for CSL code generation.

**Properties:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `output_dir` | `str` | `"./pyflame_csl_output"` | Output directory |
| `target` | `str` | `"wse2"` | Target hardware (`"wse2"` or `"wse3"`) |
| `optimize` | `bool` | `True` | Enable optimizations |
| `generate_debug_info` | `bool` | `False` | Include debug info |
| `emit_comments` | `bool` | `True` | Include source comments |
| `fabric_width` | `int` | `0` | Fabric width (0 = auto) |
| `fabric_height` | `int` | `0` | Fabric height (0 = auto) |
| `runtime_address` | `str` | `""` | Runtime endpoint address |

```python
options = pf.CodeGenOptions()
options.target = "wse2"
options.optimize = True
options.runtime_address = "localhost:9000"
```

### CodeGenResult Class

Result of code generation.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `success` | `bool` | Whether generation succeeded |
| `output_dir` | `str` | Path to output directory |
| `generated_files` | `list[str]` | List of generated file paths |
| `error_message` | `str` | Error message if failed |
| `sources` | `dict[str, str]` | Generated source code (for debugging) |

### CSLCodeGenerator Class

Generates CSL code from PyFlame graphs.

#### `generate(graph, options=CodeGenOptions())`

Generate CSL code to files.

```python
generator = pf.CSLCodeGenerator()
result = generator.generate(graph, options)
if result.success:
    print(f"Generated {len(result.generated_files)} files")
```

### `compile_to_csl(tensor, options=CodeGenOptions())`

Convenience function to compile a tensor's graph to CSL.

```python
>>> a = pf.randn([100, 100])
>>> b = pf.randn([100, 100])
>>> c = pf.relu(a @ b)
>>> result = pf.compile_to_csl(c)
```

---

## Utility Functions

### `is_lazy(tensor)`

Check if tensor has not been evaluated.

```python
>>> x = pf.randn([100])
>>> pf.is_lazy(x)
True
>>> pf.eval(x)
>>> pf.is_lazy(x)
False
```

### `eval(*tensors)`

Force evaluation of one or more tensors.

**Returns:** Single tensor if one argument, tuple if multiple.

```python
>>> a = pf.randn([100])
>>> b = pf.randn([100])
>>> c = a + b
>>> pf.eval(c)

# Evaluate multiple
>>> pf.eval(a, b, c)
```

---

## Version Information

```python
>>> import pyflame as pf
>>> pf.__version__
'1.0.0-alpha'
>>> pf.__version_info__
(1, 0, 0, 'alpha')
>>> pf.__release_status__
'Pre-Release Alpha'
```

---

## C++ API Summary

The C++ API mirrors the Python API:

```cpp
#include <pyflame/pyflame.hpp>
using namespace pyflame;

// Tensor creation
auto a = Tensor::zeros({100, 100});
auto b = Tensor::randn({100, 100}, DType::float32, MeshLayout::single_pe());

// Operations
auto c = a + b;
auto d = matmul(a, b);
auto e = relu(c);

// Reductions
auto sum = c.sum();
auto mean = c.mean(/*dim=*/0);

// Evaluation
e.eval();
float* data = e.data<float>();

// CSL generation
backend::CodeGenOptions options;
options.target = "wse2";
backend::CSLCodeGenerator generator;
auto result = generator.generate(graph, options);
```
