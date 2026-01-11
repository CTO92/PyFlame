# PyFlame API Reference

**PyFlame Version:** Pre-Release Alpha 1.0

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs described here are subject to change.

---

## Table of Contents

### Core API (Phase 1)
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

### ML Primitives (Phase 2)
14. [Automatic Differentiation](#automatic-differentiation)
15. [Neural Network Modules](#neural-network-modules)
16. [Loss Functions](#loss-functions)
17. [Optimizers](#optimizers)
18. [Learning Rate Schedulers](#learning-rate-schedulers)

### Model Support (Phase 3)
19. [Data Loading](#data-loading)
20. [Data Transforms](#data-transforms)
21. [Model Serialization](#model-serialization)
22. [Pre-built Models](#pre-built-models)
23. [Training Utilities](#training-utilities)
24. [Metrics](#metrics)
25. [Model Hub](#model-hub)

### Utilities
26. [Utility Functions](#utility-functions)

### Ecosystem (Phase 4)
27. [Developer Tools](#developer-tools)
28. [Integrations](#integrations)
29. [Model Serving](#model-serving)
30. [Benchmarking](#benchmarking)
31. [Extensions & Plugins](#extensions--plugins)

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

## Automatic Differentiation

PyFlame supports automatic differentiation (autograd) for computing gradients through the computation graph.

### GradMode

Control whether gradient computation is enabled.

```python
import pyflame as pf

# Check if gradients are enabled
pf.autograd.GradMode.is_enabled()  # True by default

# Disable gradient computation
pf.autograd.GradMode.set_enabled(False)

# Re-enable
pf.autograd.GradMode.set_enabled(True)
```

### no_grad Context Manager

Temporarily disable gradient tracking.

```python
with pf.autograd.no_grad():
    # Operations here won't track gradients
    output = model(input)
```

### backward(output, grad_output=None)

Compute gradients for the output tensor.

```python
# Forward pass
x = pf.randn([10, 5])
y = pf.randn([5, 3])
z = x @ y
loss = z.sum()

# Backward pass
pf.autograd.backward(loss)
```

---

## Neural Network Modules

The `pf.nn` module provides building blocks for neural networks.

### Module Base Class

All layers inherit from `Module`:

```python
class Module:
    def forward(self, input: Tensor) -> Tensor
    def parameters(self) -> list[Tensor]
    def zero_grad(self)
    def train(mode: bool = True)
    def eval()
    def is_training() -> bool
    def state_dict() -> dict
    def load_state_dict(dict)
```

### Linear Layer

`pf.nn.Linear(in_features, out_features, bias=True)`

Fully connected layer: y = x @ W^T + b

```python
layer = pf.nn.Linear(512, 256)
output = layer(input)  # [batch, 512] -> [batch, 256]

# Access parameters
layer.weight  # [256, 512]
layer.bias    # [256]
```

### Convolution Layers

#### Conv2d

`pf.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`

```python
conv = pf.nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
output = conv(input)  # [N, 3, H, W] -> [N, 64, H, W]
```

#### Conv1d

`pf.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`

```python
conv = pf.nn.Conv1d(128, 256, kernel_size=3, padding=1)
output = conv(input)  # [N, 128, L] -> [N, 256, L]
```

### Normalization Layers

#### BatchNorm2d

`pf.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)`

```python
bn = pf.nn.BatchNorm2d(64)
output = bn(input)  # [N, 64, H, W]
```

#### BatchNorm1d

`pf.nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)`

```python
bn = pf.nn.BatchNorm1d(256)
output = bn(input)  # [N, 256] or [N, 256, L]
```

#### LayerNorm

`pf.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`

```python
ln = pf.nn.LayerNorm([768])
output = ln(input)  # [batch, seq, 768]
```

#### GroupNorm

`pf.nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)`

```python
gn = pf.nn.GroupNorm(32, 256)  # 32 groups, 256 channels
output = gn(input)  # [N, 256, H, W]
```

### Pooling Layers

#### MaxPool2d

`pf.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False)`

```python
pool = pf.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
output = pool(input)  # [N, C, H, W] -> [N, C, H/2, W/2]
```

#### AvgPool2d

`pf.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)`

```python
pool = pf.nn.AvgPool2d(kernel_size=(2, 2))
output = pool(input)
```

#### AdaptiveAvgPool2d

`pf.nn.AdaptiveAvgPool2d(output_size)`

```python
pool = pf.nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
output = pool(input)  # [N, C, H, W] -> [N, C, 1, 1]
```

#### GlobalAvgPool2d

`pf.nn.GlobalAvgPool2d()`

```python
pool = pf.nn.GlobalAvgPool2d()
output = pool(input)  # [N, C, H, W] -> [N, C, 1, 1]
```

### Dropout Layers

#### Dropout

`pf.nn.Dropout(p=0.5, inplace=False)`

```python
dropout = pf.nn.Dropout(0.5)
dropout.train()  # Enable dropout
output = dropout(input)

dropout.eval()   # Disable dropout (inference)
output = dropout(input)  # Pass-through
```

#### Dropout2d

`pf.nn.Dropout2d(p=0.5, inplace=False)`

Drops entire channels (spatial dropout).

```python
dropout = pf.nn.Dropout2d(0.2)
output = dropout(input)  # [N, C, H, W]
```

### Attention Layers

#### MultiheadAttention

`pf.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False)`

```python
mha = pf.nn.MultiheadAttention(embed_dim=512, num_heads=8)

# Self-attention
output = mha(x)  # [seq, batch, embed] -> [seq, batch, embed]

# Cross-attention
output, attn_weights = mha.forward(query, key, value, need_weights=True)
```

#### SelfAttention

`pf.nn.SelfAttention(embed_dim, num_heads, dropout=0.0, bias=True)`

```python
attn = pf.nn.SelfAttention(768, 12)
output = attn(input)  # [seq, batch, 768]
```

#### CrossAttention

`pf.nn.CrossAttention(embed_dim, num_heads, dropout=0.0, bias=True)`

```python
attn = pf.nn.CrossAttention(512, 8)
output = attn(query, context)
```

### Sequential Container

`pf.nn.Sequential(*modules)`

```python
model = pf.nn.Sequential([
    pf.nn.Linear(784, 256),
    pf.nn.Linear(256, 128),
    pf.nn.Linear(128, 10)
])

# Or add incrementally
model = pf.nn.Sequential()
model.add(pf.nn.Linear(784, 256))
model.add(pf.nn.Linear(256, 10))

output = model(input)
```

---

## Loss Functions

The `pf.nn` module provides common loss functions.

### Reduction Modes

All loss functions support reduction modes:
- `pf.nn.Reduction.NONE` - Return per-element loss
- `pf.nn.Reduction.MEAN` - Mean of all elements (default)
- `pf.nn.Reduction.SUM` - Sum of all elements

### MSELoss

`pf.nn.MSELoss(reduction='mean')`

Mean Squared Error: L = (pred - target)²

```python
loss_fn = pf.nn.MSELoss()
loss = loss_fn(predictions, targets)
```

### L1Loss

`pf.nn.L1Loss(reduction='mean')`

Mean Absolute Error: L = |pred - target|

```python
loss_fn = pf.nn.L1Loss()
loss = loss_fn(predictions, targets)
```

### SmoothL1Loss

`pf.nn.SmoothL1Loss(reduction='mean', beta=1.0)`

Huber loss: smooth transition between L1 and L2.

```python
loss_fn = pf.nn.SmoothL1Loss(beta=1.0)
loss = loss_fn(predictions, targets)
```

### CrossEntropyLoss

`pf.nn.CrossEntropyLoss(reduction='mean', ignore_index=-100, label_smoothing=0.0)`

Combines LogSoftmax and NLLLoss for classification.

```python
loss_fn = pf.nn.CrossEntropyLoss()
# logits: [batch, num_classes], targets: [batch] (class indices)
loss = loss_fn(logits, targets)

# With label smoothing
loss_fn = pf.nn.CrossEntropyLoss(label_smoothing=0.1)
```

### BCELoss

`pf.nn.BCELoss(reduction='mean')`

Binary Cross Entropy (input should be probabilities).

```python
loss_fn = pf.nn.BCELoss()
loss = loss_fn(pf.sigmoid(logits), targets)
```

### BCEWithLogitsLoss

`pf.nn.BCEWithLogitsLoss(reduction='mean')`

Numerically stable BCE with sigmoid built-in.

```python
loss_fn = pf.nn.BCEWithLogitsLoss()
loss = loss_fn(logits, targets)  # No sigmoid needed
```

### NLLLoss

`pf.nn.NLLLoss(reduction='mean', ignore_index=-100)`

Negative Log Likelihood (input should be log-probabilities).

```python
loss_fn = pf.nn.NLLLoss()
loss = loss_fn(pf.log_softmax(logits, dim=1), targets)
```

### KLDivLoss

`pf.nn.KLDivLoss(reduction='mean', log_target=False)`

Kullback-Leibler Divergence.

```python
loss_fn = pf.nn.KLDivLoss()
loss = loss_fn(log_probs, target_probs)
```

### Functional Interface

```python
from pyflame.nn import functional as F

loss = F.mse_loss(pred, target)
loss = F.l1_loss(pred, target)
loss = F.cross_entropy(logits, targets)
loss = F.bce_loss(probs, targets)
loss = F.bce_with_logits(logits, targets)
```

---

## Optimizers

The `pf.optim` module provides optimization algorithms.

### SGD

`pf.optim.SGD(params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False)`

Stochastic Gradient Descent with optional momentum.

```python
model = pf.nn.Linear(100, 10)
optimizer = pf.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for batch in data:
    optimizer.zero_grad()
    loss = loss_fn(model(batch.x), batch.y)
    pf.autograd.backward(loss)
    optimizer.step()
```

### Adam

`pf.optim.Adam(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, amsgrad=False)`

Adam optimizer with adaptive learning rates.

```python
optimizer = pf.optim.Adam(model.parameters(), lr=0.001)

# With weight decay
optimizer = pf.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# AMSGrad variant
optimizer = pf.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
```

### AdamW

`pf.optim.AdamW(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, amsgrad=False)`

Adam with decoupled weight decay (recommended for transformers).

```python
optimizer = pf.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

### RMSprop

`pf.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False)`

```python
optimizer = pf.optim.RMSprop(model.parameters(), lr=0.01)

# With momentum
optimizer = pf.optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)
```

### Optimizer Methods

All optimizers support:

```python
optimizer.step()           # Update parameters
optimizer.zero_grad()      # Zero all gradients
optimizer.get_lr()         # Get current learning rate
optimizer.set_lr(new_lr)   # Set learning rate
optimizer.state_dict()     # Get state for checkpointing
optimizer.load_state_dict(state)  # Load state
```

---

## Learning Rate Schedulers

The `pf.optim` module provides learning rate schedulers.

### StepLR

`pf.optim.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)`

Decay LR by gamma every step_size epochs.

```python
scheduler = pf.optim.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # LR *= 0.1 at epochs 30, 60, 90
```

### MultiStepLR

`pf.optim.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)`

Decay LR at specific epochs.

```python
scheduler = pf.optim.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
# LR *= 0.1 at epochs 30 and 80
```

### ExponentialLR

`pf.optim.ExponentialLR(optimizer, gamma, last_epoch=-1)`

Decay LR by gamma every epoch.

```python
scheduler = pf.optim.ExponentialLR(optimizer, gamma=0.95)
# LR *= 0.95 each epoch
```

### CosineAnnealingLR

`pf.optim.CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1)`

Cosine annealing schedule.

```python
scheduler = pf.optim.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

### ReduceLROnPlateau

`pf.optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, cooldown=0, min_lr=0.0)`

Reduce LR when metric stops improving.

```python
scheduler = pf.optim.ReduceLROnPlateau(optimizer, mode='min', patience=10)

for epoch in range(100):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # Pass validation metric
```

### OneCycleLR

`pf.optim.OneCycleLR(optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4, last_epoch=-1)`

1cycle learning rate policy (warmup + annealing).

```python
scheduler = pf.optim.OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=1000,
    pct_start=0.3  # 30% warmup
)

for step in range(1000):
    train_step(...)
    scheduler.step()
```

### LinearLR

`pf.optim.LinearLR(optimizer, start_factor=1/3, end_factor=1.0, total_iters=5, last_epoch=-1)`

Linear learning rate warmup.

```python
scheduler = pf.optim.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
```

### PolynomialLR

`pf.optim.PolynomialLR(optimizer, total_iters, power=1.0, end_lr=0.0, last_epoch=-1)`

Polynomial decay schedule.

```python
scheduler = pf.optim.PolynomialLR(optimizer, total_iters=100, power=2.0)
```

---

## Data Loading

The `pf.data` module provides utilities for loading and batching data.

### Dataset Base Class

Abstract base class for datasets.

```python
from pyflame.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
```

### TensorDataset

`pf.data.TensorDataset(*tensors)`

Dataset wrapping tensors with the same first dimension.

```python
from pyflame.data import TensorDataset

features = np.random.randn(1000, 10)
labels = np.random.randint(0, 5, size=1000)
dataset = TensorDataset(features, labels)

x, y = dataset[0]  # Get first sample
```

### Subset

`pf.data.Subset(dataset, indices)`

Subset of a dataset at specified indices.

```python
from pyflame.data import Subset

dataset = TensorDataset(features, labels)
train_subset = Subset(dataset, range(800))
val_subset = Subset(dataset, range(800, 1000))
```

### ConcatDataset

`pf.data.ConcatDataset(datasets)`

Concatenate multiple datasets.

```python
from pyflame.data import ConcatDataset

combined = ConcatDataset([dataset1, dataset2])
# Or use + operator
combined = dataset1 + dataset2
```

### MapDataset

`pf.data.MapDataset(dataset, transform)`

Apply a transform function to dataset items.

```python
from pyflame.data import MapDataset

mapped = MapDataset(dataset, lambda x: x * 2)
```

### random_split

`pf.data.random_split(dataset, lengths, generator=None)`

Randomly split a dataset into non-overlapping subsets.

**Parameters:**
- `dataset`: Dataset to split
- `lengths`: List of lengths or fractions (must sum to dataset length or 1.0)
- `generator`: Random number generator for reproducibility

```python
from pyflame.data import random_split

# Split by lengths
train, val, test = random_split(dataset, [800, 100, 100])

# Split by fractions
train, val, test = random_split(dataset, [0.8, 0.1, 0.1])
```

### DataLoader

`pf.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, drop_last=False, pin_memory=False)`

Combines a dataset and sampler, providing an iterable over batches.

**Parameters:**
- `dataset`: Dataset to load from
- `batch_size`: Samples per batch
- `shuffle`: Shuffle data each epoch
- `sampler`: Custom sampler (mutually exclusive with shuffle)
- `batch_sampler`: Custom batch sampler
- `num_workers`: Subprocesses for loading (0 = main process)
- `collate_fn`: Function to merge samples into batch
- `drop_last`: Drop incomplete final batch
- `pin_memory`: Copy tensors to pinned memory

```python
from pyflame.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch in loader:
    x, y = batch
    output = model(x)
```

### Samplers

#### SequentialSampler

`pf.data.SequentialSampler(data_source)`

Samples elements sequentially.

```python
from pyflame.data import SequentialSampler

sampler = SequentialSampler(dataset)
```

#### RandomSampler

`pf.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)`

Samples elements randomly.

```python
from pyflame.data import RandomSampler

sampler = RandomSampler(dataset)

# With replacement (for oversampling)
sampler = RandomSampler(dataset, replacement=True, num_samples=2000)
```

#### SubsetRandomSampler

`pf.data.SubsetRandomSampler(indices, generator=None)`

Samples randomly from specified indices.

```python
from pyflame.data import SubsetRandomSampler

indices = [0, 2, 4, 6, 8]
sampler = SubsetRandomSampler(indices)
```

#### WeightedRandomSampler

`pf.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)`

Samples according to weights.

```python
from pyflame.data import WeightedRandomSampler

# Higher weights = more likely to be sampled
weights = [1.0, 2.0, 3.0, 4.0]
sampler = WeightedRandomSampler(weights, num_samples=100)
```

#### BatchSampler

`pf.data.BatchSampler(sampler, batch_size, drop_last=False)`

Wraps another sampler to yield batches of indices.

```python
from pyflame.data import BatchSampler, SequentialSampler

sampler = SequentialSampler(dataset)
batch_sampler = BatchSampler(sampler, batch_size=32, drop_last=True)
```

#### DistributedSampler

`pf.data.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)`

Sampler for distributed training.

```python
from pyflame.data import DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

## Data Transforms

The `pf.data.transforms` module provides data transformation utilities.

### Compose

`pf.data.transforms.Compose(transforms)`

Chain multiple transforms together.

```python
from pyflame.data.transforms import Compose, Normalize, ToTensor

transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = transform(raw_data)
```

### Lambda

`pf.data.transforms.Lambda(func)`

Apply a custom function as transform.

```python
from pyflame.data.transforms import Lambda

transform = Lambda(lambda x: x ** 2)
```

### ToTensor

`pf.data.transforms.ToTensor()`

Convert numpy array to tensor-compatible format.

```python
from pyflame.data.transforms import ToTensor

transform = ToTensor()
tensor_data = transform(numpy_array)
```

### Normalize

`pf.data.transforms.Normalize(mean, std)`

Normalize with mean and standard deviation.

```python
from pyflame.data.transforms import Normalize

# ImageNet normalization
transform = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Image Transforms

#### Resize

`pf.data.transforms.Resize(size, interpolation='bilinear')`

```python
from pyflame.data.transforms import Resize

transform = Resize((224, 224))
```

#### CenterCrop

`pf.data.transforms.CenterCrop(size)`

```python
from pyflame.data.transforms import CenterCrop

transform = CenterCrop((224, 224))
```

#### RandomCrop

`pf.data.transforms.RandomCrop(size, padding=None, pad_if_needed=False)`

```python
from pyflame.data.transforms import RandomCrop

transform = RandomCrop((224, 224), padding=4)
```

#### RandomHorizontalFlip

`pf.data.transforms.RandomHorizontalFlip(p=0.5)`

```python
from pyflame.data.transforms import RandomHorizontalFlip

transform = RandomHorizontalFlip(p=0.5)
```

#### RandomVerticalFlip

`pf.data.transforms.RandomVerticalFlip(p=0.5)`

```python
from pyflame.data.transforms import RandomVerticalFlip

transform = RandomVerticalFlip(p=0.5)
```

#### RandomRotation

`pf.data.transforms.RandomRotation(degrees, expand=False, center=None)`

```python
from pyflame.data.transforms import RandomRotation

transform = RandomRotation(degrees=15)
```

#### ColorJitter

`pf.data.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`

```python
from pyflame.data.transforms import ColorJitter

transform = ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1
)
```

### Composition Transforms

#### RandomApply

`pf.data.transforms.RandomApply(transforms, p=0.5)`

Apply transforms with probability p.

```python
from pyflame.data.transforms import RandomApply, ColorJitter

transform = RandomApply([ColorJitter(0.2, 0.2)], p=0.3)
```

#### RandomChoice

`pf.data.transforms.RandomChoice(transforms)`

Randomly select one transform to apply.

```python
from pyflame.data.transforms import RandomChoice, Resize

transform = RandomChoice([
    Resize((224, 224)),
    Resize((256, 256)),
])
```

#### RandomOrder

`pf.data.transforms.RandomOrder(transforms)`

Apply transforms in random order.

```python
from pyflame.data.transforms import RandomOrder

transform = RandomOrder([transform1, transform2, transform3])
```

---

## Model Serialization

PyFlame supports multiple serialization formats for saving and loading models.

### save

`pf.save(obj, path, format='auto')`

Save model or state dict to file.

**Parameters:**
- `obj`: Model, state dict, or tensor to save
- `path`: File path
- `format`: `'auto'`, `'pyflame'`, `'safetensors'`, `'numpy'`

```python
import pyflame as pf

# Save model state dict
pf.save(model.state_dict(), "model.pf")

# Save with specific format
pf.save(model.state_dict(), "model.safetensors", format='safetensors')

# Save entire model (architecture + weights)
pf.save(model, "full_model.pf")
```

### load

`pf.load(path, format='auto', device=None)`

Load model or state dict from file.

**Parameters:**
- `path`: File path
- `format`: `'auto'` (detect from extension), `'pyflame'`, `'safetensors'`, `'numpy'`
- `device`: Target device for loaded tensors

```python
import pyflame as pf

# Load state dict
state_dict = pf.load("model.pf")
model.load_state_dict(state_dict)

# Load with format detection
weights = pf.load("model.safetensors")
```

### Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| PyFlame Native | `.pf` | Native binary format (default) |
| SafeTensors | `.safetensors` | Safe, fast tensor serialization |
| NumPy | `.npz` | NumPy compressed archive |

### Model Checkpointing

```python
# Save checkpoint with optimizer state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
pf.save(checkpoint, f'checkpoint_epoch_{epoch}.pf')

# Load checkpoint
checkpoint = pf.load('checkpoint_epoch_10.pf')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## Pre-built Models

PyFlame provides pre-built model architectures.

### ResNet

#### ResNet Variants

```python
from pyflame.models import resnet18, resnet34, resnet50, resnet101, resnet152

# Basic usage
model = resnet50(num_classes=1000)

# Custom configuration
model = resnet50(
    num_classes=100,
    pretrained=True,
    zero_init_residual=True
)
```

#### ResNetConfig

```python
from pyflame.models import ResNetConfig, ResNet

config = ResNetConfig(
    num_classes=1000,
    in_channels=3,
    block_type='bottleneck',  # 'basic' or 'bottleneck'
    layers=[3, 4, 6, 3],      # Blocks per stage
    base_width=64,
    groups=1,
    width_per_group=64,
    replace_stride_with_dilation=[False, False, False],
    norm_layer='batch_norm',
    zero_init_residual=False
)

model = ResNet(config)
```

#### ResNeXt

```python
from pyflame.models import resnext50_32x4d, resnext101_32x8d

model = resnext50_32x4d(num_classes=1000)  # 32 groups, 4 width per group
model = resnext101_32x8d(num_classes=1000)  # 32 groups, 8 width per group
```

#### Wide ResNet

```python
from pyflame.models import wide_resnet50_2, wide_resnet101_2

model = wide_resnet50_2(num_classes=1000)  # 2x wider
```

### Transformer / BERT

#### TransformerEncoderLayer

```python
from pyflame.models import TransformerEncoderLayer

layer = TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    batch_first=True,
    norm_first=False
)

output = layer(src, src_mask=None, src_key_padding_mask=None)
```

#### TransformerDecoderLayer

```python
from pyflame.models import TransformerDecoderLayer

layer = TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    batch_first=True
)

output = layer(tgt, memory, tgt_mask=None, memory_mask=None)
```

#### TransformerEncoder

```python
from pyflame.models import TransformerEncoder, TransformerEncoderLayer

encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
encoder = TransformerEncoder(encoder_layer, num_layers=6)

output = encoder(src)
```

#### BertModel

```python
from pyflame.models import BertConfig, BertModel

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    layer_norm_eps=1e-12
)

model = BertModel(config)
output = model(input_ids, attention_mask=None, token_type_ids=None)
```

#### BertForSequenceClassification

```python
from pyflame.models import BertConfig, BertForSequenceClassification

config = BertConfig(vocab_size=30522, num_classes=2)
model = BertForSequenceClassification(config)

logits = model(input_ids, attention_mask=attention_mask)
```

---

## Training Utilities

The `pf.training` module provides high-level training utilities.

### Trainer

`pf.training.Trainer(model, optimizer, loss_fn, config=None, callbacks=None, metrics=None)`

High-level training interface.

**Parameters:**
- `model`: Model to train
- `optimizer`: Optimizer instance
- `loss_fn`: Loss function
- `config`: TrainerConfig instance
- `callbacks`: List of callback instances
- `metrics`: List or dict of metrics

```python
from pyflame.training import Trainer, TrainerConfig

config = TrainerConfig(
    max_epochs=100,
    grad_clip_norm=1.0,
    log_interval=10,
    checkpoint_dir='./checkpoints',
    early_stopping_patience=10
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=pf.nn.CrossEntropyLoss(),
    config=config,
    callbacks=[EarlyStopping(patience=5)],
    metrics={'accuracy': Accuracy()}
)
```

#### TrainerConfig

```python
from pyflame.training import TrainerConfig

config = TrainerConfig(
    max_epochs=100,
    max_steps=None,              # Max training steps (overrides epochs)
    grad_clip_norm=None,         # Gradient clipping norm
    grad_clip_value=None,        # Gradient clipping value
    accumulate_grad_batches=1,   # Gradient accumulation
    log_interval=10,             # Steps between logging
    eval_interval=None,          # Steps between evaluation
    checkpoint_dir=None,         # Directory for checkpoints
    checkpoint_interval=1,       # Epochs between checkpoints
    early_stopping_patience=None,
    early_stopping_metric='val_loss',
    early_stopping_mode='min'
)
```

#### Trainer Methods

```python
# Train model
history = trainer.fit(
    train_loader,
    val_loader=None,
    epochs=None  # Override config
)

# Evaluate model
metrics = trainer.evaluate(val_loader)

# Run test
test_metrics = trainer.test(test_loader)

# Get predictions
predictions = trainer.predict(data_loader)

# Save/load checkpoints
trainer.save_checkpoint("checkpoint.pf")
trainer.load_checkpoint("checkpoint.pf")
```

### Callbacks

#### EarlyStopping

`pf.training.EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0, restore_best_weights=True)`

Stop training when metric stops improving.

```python
from pyflame.training.callbacks import EarlyStopping

callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    restore_best_weights=True
)
```

#### ModelCheckpoint

`pf.training.ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False)`

Save model checkpoints during training.

```python
from pyflame.training.callbacks import ModelCheckpoint

callback = ModelCheckpoint(
    filepath='./checkpoints/model_{epoch:02d}_{val_loss:.4f}.pf',
    monitor='val_loss',
    save_best_only=True
)
```

#### LearningRateScheduler

`pf.training.LearningRateScheduler(scheduler, monitor=None)`

Adjust learning rate during training.

```python
from pyflame.training.callbacks import LearningRateScheduler

scheduler = pf.optim.CosineAnnealingLR(optimizer, T_max=100)
callback = LearningRateScheduler(scheduler)
```

#### ProgressBar

`pf.training.ProgressBar()`

Display training progress bar.

```python
from pyflame.training.callbacks import ProgressBar

callback = ProgressBar()
```

#### CSVLogger

`pf.training.CSVLogger(filepath, separator=',', append=False)`

Log training metrics to CSV file.

```python
from pyflame.training.callbacks import CSVLogger

callback = CSVLogger('./logs/training.csv')
```

#### TensorBoardLogger

`pf.training.TensorBoardLogger(log_dir)`

Log metrics to TensorBoard.

```python
from pyflame.training.callbacks import TensorBoardLogger

callback = TensorBoardLogger('./logs/tensorboard')
```

#### Custom Callbacks

```python
from pyflame.training.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} - loss: {logs.get('loss'):.4f}")

    def on_batch_end(self, batch, logs=None):
        pass  # Called after each batch
```

---

## Metrics

The `pf.metrics` module provides evaluation metrics.

### Classification Metrics

#### Accuracy

`pf.metrics.Accuracy(task='multiclass', num_classes=None, top_k=1, threshold=0.5)`

```python
from pyflame.metrics import Accuracy

metric = Accuracy(task='multiclass', num_classes=10)
metric.update(predictions, targets)
acc = metric.compute()
metric.reset()
```

#### Precision

`pf.metrics.Precision(task='binary', num_classes=None, average='macro', threshold=0.5)`

```python
from pyflame.metrics import Precision

metric = Precision(task='multiclass', num_classes=10, average='macro')
metric.update(predictions, targets)
precision = metric.compute()
```

#### Recall

`pf.metrics.Recall(task='binary', num_classes=None, average='macro', threshold=0.5)`

```python
from pyflame.metrics import Recall

metric = Recall(task='multiclass', num_classes=10, average='weighted')
metric.update(predictions, targets)
recall = metric.compute()
```

#### F1Score

`pf.metrics.F1Score(task='binary', num_classes=None, average='macro', threshold=0.5)`

```python
from pyflame.metrics import F1Score

metric = F1Score(task='multiclass', num_classes=10, average='macro')
metric.update(predictions, targets)
f1 = metric.compute()
```

#### ConfusionMatrix

`pf.metrics.ConfusionMatrix(num_classes, normalize=None)`

```python
from pyflame.metrics import ConfusionMatrix

metric = ConfusionMatrix(num_classes=10)
metric.update(predictions, targets)
matrix = metric.compute()
```

#### AUROC

`pf.metrics.AUROC(task='binary', num_classes=None, average='macro')`

Area Under ROC Curve.

```python
from pyflame.metrics import AUROC

metric = AUROC(task='multiclass', num_classes=10)
metric.update(probabilities, targets)
auroc = metric.compute()
```

#### AveragePrecision

`pf.metrics.AveragePrecision(task='binary', num_classes=None, average='macro')`

```python
from pyflame.metrics import AveragePrecision

metric = AveragePrecision(task='binary')
metric.update(probabilities, targets)
ap = metric.compute()
```

### Regression Metrics

#### MeanSquaredError

`pf.metrics.MeanSquaredError(squared=True)`

```python
from pyflame.metrics import MeanSquaredError

metric = MeanSquaredError()
metric.update(predictions, targets)
mse = metric.compute()
```

#### RootMeanSquaredError

`pf.metrics.RootMeanSquaredError()`

```python
from pyflame.metrics import RootMeanSquaredError

metric = RootMeanSquaredError()
metric.update(predictions, targets)
rmse = metric.compute()
```

#### MeanAbsoluteError

`pf.metrics.MeanAbsoluteError()`

```python
from pyflame.metrics import MeanAbsoluteError

metric = MeanAbsoluteError()
metric.update(predictions, targets)
mae = metric.compute()
```

#### R2Score

`pf.metrics.R2Score()`

Coefficient of determination.

```python
from pyflame.metrics import R2Score

metric = R2Score()
metric.update(predictions, targets)
r2 = metric.compute()
```

#### MeanAbsolutePercentageError

`pf.metrics.MeanAbsolutePercentageError()`

```python
from pyflame.metrics import MeanAbsolutePercentageError

metric = MeanAbsolutePercentageError()
metric.update(predictions, targets)
mape = metric.compute()
```

### MetricCollection

`pf.metrics.MetricCollection(metrics)`

Group multiple metrics together.

```python
from pyflame.metrics import MetricCollection, Accuracy, Precision, Recall

metrics = MetricCollection({
    'accuracy': Accuracy(),
    'precision': Precision(),
    'recall': Recall()
})

metrics.update(predictions, targets)
results = metrics.compute()  # {'accuracy': 0.95, 'precision': 0.93, 'recall': 0.92}
metrics.reset()
```

---

## Model Hub

The `pf.hub` module provides model registry and pretrained weights management.

### Model Registry

#### Registering Models

```python
from pyflame.hub import register_model

@register_model(
    name="my_resnet",
    description="Custom ResNet variant",
    tags=["vision", "classification"],
    pretrained_weights=["imagenet1k"]
)
def create_my_resnet(**kwargs):
    return MyResNet(**kwargs)
```

#### Getting Models

```python
from pyflame.hub import get_model

# Get model without pretrained weights
model = get_model("resnet50", num_classes=100)

# Get model with pretrained weights
model = get_model("resnet50", pretrained=True)

# Get specific pretrained weights
model = get_model("resnet50", pretrained="imagenet1k-v2")
```

#### Listing Models

```python
from pyflame.hub import list_models, model_info

# List all registered models
all_models = list_models()

# List models with specific tag
vision_models = list_models(tag="vision")

# Get model information
info = model_info("resnet50")
print(info.description)
print(info.pretrained_weights)
```

### Pretrained Weights

#### Loading Pretrained Weights

```python
from pyflame.hub import load_pretrained

model = ResNet50()
load_pretrained(model, "resnet50", "imagenet1k-v1")
```

#### Downloading Weights

```python
from pyflame.hub import download_weights, get_weight_path

# Download weights (cached automatically)
path = download_weights("resnet50", "imagenet1k-v1")

# Get path to cached weights
path = get_weight_path("resnet50", "imagenet1k-v1")
```

#### Listing Available Weights

```python
from pyflame.hub import list_pretrained, weight_info

# List all pretrained weights
all_weights = list_pretrained()

# List weights for specific model
resnet_weights = list_pretrained("resnet50")

# Get weight information
info = weight_info("resnet50", "imagenet1k-v1")
print(f"Size: {info.size_mb} MB")
print(f"Accuracy: {info.metrics['top1_accuracy']}%")
```

#### Cache Management

```python
from pyflame.hub import cache_info, clear_cache

# Get cache information
info = cache_info()
print(f"Cache directory: {info['cache_dir']}")
print(f"Total size: {info['size_mb']:.1f} MB")
print(f"Cached models: {list(info['cached_models'].keys())}")

# Clear cache for specific model
clear_cache("resnet50")

# Clear all cached weights
clear_cache()
```

### Available Pretrained Models

| Model | Weights | Dataset | Top-1 Accuracy |
|-------|---------|---------|----------------|
| resnet18 | imagenet1k-v1 | ImageNet-1K | 69.76% |
| resnet50 | imagenet1k-v1 | ImageNet-1K | 76.13% |
| resnet50 | imagenet1k-v2 | ImageNet-1K | 80.86% |
| bert-base-uncased | default | Wikipedia + BookCorpus | - |
| vit-base-patch16-224 | imagenet1k | ImageNet-1K | - |
| gpt2 | default | WebText | - |

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

---

## Developer Tools

The `pf.tools` module provides debugging, profiling, and visualization capabilities.

### PyFlameDebugger

Interactive debugger for PyFlame computations.

```python
from pyflame.tools import PyFlameDebugger

# Create debugger
debugger = PyFlameDebugger()

# Set breakpoints
debugger.set_breakpoint("matmul", condition="output.shape[0] > 1000")
debugger.set_breakpoint("relu")

# Watch tensors
debugger.watch("hidden", lambda t: t.mean())

# Use as context manager
with debugger:
    output = model(input)
```

#### Breakpoint Methods

```python
# Set breakpoint on operation
debugger.set_breakpoint(op_name, condition=None, callback=None)

# Remove breakpoint
debugger.remove_breakpoint(op_name)

# Enable/disable breakpoint
debugger.enable_breakpoint(op_name)
debugger.disable_breakpoint(op_name)

# Clear all breakpoints
debugger.clear_breakpoints()

# List breakpoints
breakpoints = debugger.list_breakpoints()
```

#### Watch Methods

```python
# Watch a tensor by name
debugger.watch(name, callback=None)

# Unwatch tensor
debugger.unwatch(name)

# Clear all watches
debugger.clear_watches()
```

#### Global Functions

```python
from pyflame.tools import set_breakpoint, remove_breakpoint, clear_breakpoints

set_breakpoint("relu")
remove_breakpoint("relu")
clear_breakpoints()
```

### Profiler

Performance profiling for PyFlame operations.

```python
from pyflame.tools import Profiler

# Create profiler
profiler = Profiler(track_memory=True, track_cuda_events=False)

# Profile a block of code
with profiler:
    output = model(input)

# Get results
result = profiler.get_result()
print(result.summary())

# Export to Chrome trace format
profiler.export_chrome_trace("profile.json")
```

#### ProfileResult

```python
result = profiler.get_result()

# Get summary string
print(result.summary())

# Access events
for event in result.events:
    print(f"{event.name}: {event.duration_ms:.2f}ms")

# Total time
print(f"Total: {result.total_time_ms:.2f}ms")

# Memory stats (if track_memory=True)
print(f"Peak memory: {result.peak_memory_mb:.1f}MB")
```

#### Profile Decorator

```python
from pyflame.tools import profile

@profile
def my_function():
    # Function will be profiled
    return model(input)

# With custom profiler
@profile(profiler=my_profiler)
def my_function():
    pass
```

#### Global Profiler Context

```python
from pyflame.tools import enable_profiling, disable_profiling, get_profiler

# Enable global profiling
enable_profiling()

# Run code (automatically profiled)
output = model(input)

# Disable and get results
disable_profiling()
profiler = get_profiler()
print(profiler.get_result().summary())
```

### GraphVisualizer

Visualize PyFlame computation graphs.

```python
from pyflame.tools import GraphVisualizer, visualize_graph

# Create visualizer
viz = GraphVisualizer(
    graph,
    show_shapes=True,
    show_dtypes=True,
    show_values=False,
    max_nodes=500,
    rankdir="TB"  # TB=top-bottom, LR=left-right
)

# Export to different formats
viz.to_dot("graph.dot")      # DOT format
viz.to_svg("graph.svg")      # SVG (requires graphviz)
viz.to_png("graph.png")      # PNG (requires graphviz)
viz.to_html("graph.html")    # Interactive HTML
```

#### Convenience Functions

```python
from pyflame.tools import visualize_graph, visualize_model

# Visualize a graph
visualize_graph(graph, "output.svg", format="svg")

# Visualize a model by tracing
visualize_model(model, example_input, "model.svg")
```

---

## Integrations

The `pf.integrations` module provides interoperability with ML ecosystem tools.

### ONNX Export/Import

```python
from pyflame.integrations import ONNXExporter, ONNXImporter

# Export model to ONNX
exporter = ONNXExporter(opset_version=17)
exporter.export(
    model,
    example_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}
)

# Import ONNX model
importer = ONNXImporter()
model = importer.import_model("model.onnx")
```

#### ONNXExporter Options

```python
exporter = ONNXExporter(
    opset_version=17,        # ONNX opset version
    do_constant_folding=True, # Fold constants
    verbose=False            # Print export info
)

# Verify exported model
exporter.verify(model, example_input, "model.onnx")
```

#### Convenience Functions

```python
from pyflame.integrations import export_onnx, import_onnx

# Quick export
export_onnx(model, example_input, "model.onnx")

# Quick import
model = import_onnx("model.onnx")
```

### Weights & Biases Integration

```python
from pyflame.integrations import WandbCallback

# Create callback for training
callback = WandbCallback(
    project="my-project",
    name="experiment-1",
    config={"lr": 0.001, "batch_size": 32},
    log_model=True,
    log_gradients=False
)

# Use with Trainer
trainer = Trainer(model, optimizer, loss_fn, callbacks=[callback])
trainer.fit(train_loader)

# Or use manually
callback.on_train_begin()
for epoch in range(epochs):
    callback.on_epoch_begin(epoch)
    # ... training ...
    callback.on_epoch_end(epoch, logs={"loss": loss, "accuracy": acc})
callback.on_train_end()
```

#### WandbCallback Methods

```python
# Log custom metrics
callback.log({"custom_metric": value})

# Log images
callback.log_image("predictions", image_array)

# Log model artifact
callback.log_model(model, "model_checkpoint")

# Finish run
callback.finish()
```

### MLflow Integration

```python
from pyflame.integrations import MLflowCallback

# Create callback
callback = MLflowCallback(
    experiment_name="my-experiment",
    run_name="run-1",
    tracking_uri="http://localhost:5000",
    log_models=True
)

# Use with Trainer
trainer = Trainer(model, optimizer, loss_fn, callbacks=[callback])
trainer.fit(train_loader)
```

#### MLflowCallback Methods

```python
# Log parameters
callback.log_params({"lr": 0.001})

# Log metrics
callback.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)

# Log artifact
callback.log_artifact("model.pf")

# Log model
callback.log_model(model, "model")

# End run
callback.end_run()
```

### Jupyter Integration

```python
from pyflame.integrations import setup_jupyter

# Enable rich display in Jupyter
setup_jupyter()

# Now tensors display nicely
x = pf.randn([3, 4])
x  # Shows formatted tensor with shape, dtype, stats
```

#### Jupyter Display Options

```python
from pyflame.integrations import JupyterConfig

# Configure display
JupyterConfig.max_rows = 10
JupyterConfig.max_cols = 10
JupyterConfig.precision = 4
JupyterConfig.show_dtype = True
JupyterConfig.show_shape = True
```

---

## Model Serving

The `pf.serving` module provides inference optimization and deployment.

### InferenceEngine

Optimized inference engine with caching and batching.

```python
from pyflame.serving import InferenceEngine, InferenceConfig

# Create config
config = InferenceConfig(
    batch_size=32,
    max_batch_size=64,
    enable_caching=True,
    cache_size=1000,
    warmup_iterations=10
)

# Create engine
engine = InferenceEngine(model, config)

# Warmup
engine.warmup(example_input)

# Run inference
output = engine.infer(input_data)

# Get with timing
output, latency_ms = engine.infer_with_time(input_data)

# Get statistics
stats = engine.get_stats()
print(f"Requests: {stats.total_requests}")
print(f"Avg latency: {stats.average_time_ms:.2f}ms")
print(f"Cache hit rate: {stats.cache_hit_rate:.1%}")
```

#### InferenceConfig Options

```python
config = InferenceConfig(
    batch_size=1,           # Default batch size
    max_batch_size=32,      # Maximum batch size
    enable_caching=True,    # Enable result caching
    cache_size=1000,        # LRU cache size
    warmup_iterations=10,   # Warmup iterations
    enable_fp16=False,      # Use FP16 inference
    num_threads=None        # Thread count (None = auto)
)
```

#### InferenceStats

```python
stats = engine.get_stats()

stats.total_requests      # Total inference requests
stats.total_time_ms       # Total time spent
stats.average_time_ms     # Average latency
stats.throughput          # Requests per second
stats.cache_hits          # Cache hits
stats.cache_misses        # Cache misses
stats.cache_hit_rate      # Hit rate (0.0 - 1.0)

# Reset statistics
engine.reset_stats()

# Get engine info
info = engine.get_info()
print(info)  # Model info, config, stats
```

### ModelServer

REST API server for model inference.

```python
from pyflame.serving import ModelServer, ServerConfig

# Create server config
config = ServerConfig(
    host="0.0.0.0",
    port=8000,
    workers=4,
    max_batch_size=32,
    timeout=30.0
)

# Create and start server
server = ModelServer(model, config)
server.start()  # Blocking

# Or run in background
server.start_background()
# ... do other work ...
server.stop()
```

#### ServerConfig Options

```python
config = ServerConfig(
    host="0.0.0.0",         # Bind host
    port=8000,              # Bind port
    workers=1,              # Worker processes
    max_batch_size=32,      # Max batch size
    timeout=30.0,           # Request timeout
    enable_cors=True,       # Enable CORS
    api_key=None,           # Optional API key
    ssl_cert=None,          # SSL certificate path
    ssl_key=None            # SSL key path
)
```

#### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/predict` | POST | Run inference |
| `/v1/health` | GET | Health check |
| `/v1/info` | GET | Model info |
| `/v1/stats` | GET | Server statistics |

#### Request Format

```python
# POST /v1/predict
{
    "inputs": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "batch_size": 2
}

# Response
{
    "outputs": [[0.1, 0.9], [0.8, 0.2]],
    "latency_ms": 5.2
}
```

### ModelClient

Client for calling ModelServer.

```python
from pyflame.serving import ModelClient, ClientConfig

# Create client
config = ClientConfig(
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0
)
client = ModelClient("http://localhost:8000", config)

# Run inference
output = client.predict(input_data)

# Batch inference
outputs = client.predict_batch([input1, input2, input3])

# Health check
is_healthy = client.health_check()

# Get server info
info = client.get_info()
```

#### Async Client

```python
import asyncio

async def main():
    async with ModelClient("http://localhost:8000") as client:
        output = await client.predict_async(input_data)
        outputs = await client.predict_batch_async(inputs)

asyncio.run(main())
```

### Optimization Functions

```python
from pyflame.serving import optimize_for_inference

# Optimize model for inference
optimized_model = optimize_for_inference(
    model,
    example_input,
    enable_fusion=True,       # Fuse operations
    enable_quantization=False, # Quantize weights
    target_device="cpu"       # Target device
)
```

---

## Benchmarking

The `pf.benchmarks` module provides performance benchmarking utilities.

### BenchmarkRunner

```python
from pyflame.benchmarks import BenchmarkRunner, BenchmarkConfig

# Create config
config = BenchmarkConfig(
    warmup_iterations=10,
    benchmark_iterations=100,
    batch_sizes=[1, 8, 32, 64],
    device="cpu"
)

# Create runner
runner = BenchmarkRunner(config)

# Benchmark a model
results = runner.run_model_benchmark(
    "resnet50",
    model,
    input_shape=[3, 224, 224]
)

# Print results
for result in results:
    print(f"Batch {result.batch_size}: {result.latency_ms:.2f}ms, "
          f"{result.throughput:.0f} samples/sec")
```

#### BenchmarkConfig Options

```python
config = BenchmarkConfig(
    warmup_iterations=10,      # Warmup iterations
    benchmark_iterations=100,  # Benchmark iterations
    batch_sizes=[1, 8, 32],    # Batch sizes to test
    device="cpu",              # Device to benchmark on
    sync_cuda=True,            # Sync CUDA after each iteration
    enable_memory_tracking=True # Track memory usage
)
```

#### BenchmarkResult

```python
result.name           # Benchmark name
result.batch_size     # Batch size
result.latency_ms     # Average latency (ms)
result.std_dev_ms     # Standard deviation
result.throughput     # Samples per second
result.memory_mb      # Memory usage (MB)

# Convert to dict
result.to_dict()
```

### Operation Benchmarking

```python
# Benchmark a specific operation
result = runner.run_operation_benchmark(
    "matmul_1024x1024",
    lambda x, y: x @ y,
    input_shapes=[[1024, 1024], [1024, 1024]]
)

print(f"Latency: {result.latency_ms:.2f}ms")
```

### Export Results

```python
# Export to JSON
runner.export_json("benchmark_results.json")

# Export to CSV
runner.export_csv("benchmark_results.csv")

# Clear results
runner.clear_results()
```

### Benchmark Convenience Function

```python
from pyflame.benchmarks import benchmark

# Quick benchmark
results = benchmark(
    model,
    input_shape=[3, 224, 224],
    batch_sizes=[1, 8, 32],
    iterations=100,
    print_results=True
)
```

### BenchmarkReport

```python
from pyflame.benchmarks import BenchmarkReport

# Create report from results
report = BenchmarkReport(
    name="ResNet50 Benchmark",
    results=runner.results,
    metadata={"device": "CPU", "framework": "PyFlame"}
)

# Get summary
print(report.summary())

# Export
report.save_json("report.json")
report.save_html("report.html")

# Load report
loaded = BenchmarkReport.load_json("report.json")
```

### Compare Results

```python
from pyflame.benchmarks import compare_results

# Compare two benchmark runs
comparison = compare_results(baseline_results, new_results)

for name, diff in comparison.items():
    print(f"{name}: {diff['speedup']:.2f}x speedup")
```

### Standard Benchmark Models

```python
from pyflame.benchmarks import (
    get_benchmark_model,
    list_benchmark_models,
    get_benchmark_model_info
)

# List available models
models = list_benchmark_models()
for info in models:
    print(f"{info.name}: {info.description}")

# Filter by category
vision_models = list_benchmark_models(category="vision")
nlp_models = list_benchmark_models(category="nlp")

# Get model info
info = get_benchmark_model_info("resnet50")
print(f"Params: {info.num_params:,}")
print(f"FLOPs: {info.flops:,}")

# Get model instance
model = get_benchmark_model("resnet50", num_classes=1000)
```

#### Available Benchmark Models

| Model | Category | Params | Input Shape |
|-------|----------|--------|-------------|
| `resnet18` | vision | 11.7M | [3, 224, 224] |
| `resnet50` | vision | 25.6M | [3, 224, 224] |
| `bert_base` | nlp | 110M | [512] |
| `mlp_small` | general | 100K | [128] |
| `mlp_large` | general | 5M | [512] |
| `conv_small` | vision | 50K | [3, 32, 32] |
| `transformer_encoder` | nlp | 20M | [128, 512] |

### Timing Context Manager

```python
from pyflame.benchmarks import timed

with timed() as t:
    output = model(input)

print(f"Elapsed: {t.elapsed_ms:.2f}ms")
```

---

## Extensions & Plugins

The `pf.extend` module provides extensibility mechanisms.

### Custom Operators

Register custom operations with PyFlame.

```python
from pyflame.extend import register_custom_op, custom_op

# Register a custom operation
@custom_op(name="my_activation")
def my_activation(x):
    return x * pf.sigmoid(x)  # Swish-like

# Use it
output = my_activation(input)
```

#### With Backward Pass

```python
@custom_op(name="my_relu", backward=my_relu_backward)
def my_relu(x):
    return pf.maximum(x, 0)

def my_relu_backward(grad_output, x):
    return grad_output * (x > 0)
```

#### Functional Registration

```python
from pyflame.extend import register_custom_op

def custom_norm(x, eps=1e-5):
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / pf.sqrt(var + eps)

register_custom_op(
    name="custom_norm",
    forward_fn=custom_norm,
    backward_fn=custom_norm_backward,  # Optional
    schema={"eps": float}              # Optional schema
)
```

#### Manage Custom Ops

```python
from pyflame.extend import (
    get_custom_op,
    list_custom_ops,
    unregister_custom_op,
    clear_custom_ops
)

# Get registered op
op = get_custom_op("my_activation")

# List all custom ops
ops = list_custom_ops()

# Unregister
unregister_custom_op("my_activation")

# Clear all
clear_custom_ops()
```

### Autograd Functions

Create custom autograd functions for more control.

```python
from pyflame.extend import AutogradFunction, FunctionContext

class MyFunction(AutogradFunction):
    @staticmethod
    def forward(ctx: FunctionContext, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: FunctionContext, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x

# Use it
output = MyFunction.apply(x, y)
```

### Plugins

Create reusable plugin packages.

```python
from pyflame.extend import Plugin, PluginInfo, register_plugin

class MyPlugin(Plugin):
    @classmethod
    def get_info(cls) -> PluginInfo:
        return PluginInfo(
            name="my-plugin",
            version="1.0.0",
            author="Your Name",
            description="My custom plugin"
        )

    def on_load(self):
        # Called when plugin is loaded
        print("Plugin loaded!")
        # Register custom ops, callbacks, etc.

    def on_unload(self):
        # Called when plugin is unloaded
        print("Plugin unloaded!")

    def get_custom_ops(self):
        # Return dict of custom operations
        return {
            "my_op": my_op_function
        }

# Register the plugin
register_plugin(MyPlugin)
```

### Plugin Manager

```python
from pyflame.extend import PluginManager

# Get plugin manager (singleton)
manager = PluginManager()

# Load a plugin
manager.load_plugin("my-plugin")

# Get loaded plugin
plugin = manager.get_plugin("my-plugin")

# List loaded plugins
plugins = manager.list_plugins()

# Unload plugin
manager.unload_plugin("my-plugin")

# Discover plugins from directory
manager.discover_plugins("./plugins")
```

#### Global Plugin Functions

```python
from pyflame.extend import (
    load_plugin,
    unload_plugin,
    get_plugin,
    list_plugins
)

# Load plugin by name
load_plugin("my-plugin")

# Get plugin instance
plugin = get_plugin("my-plugin")

# List all plugins
plugins = list_plugins()

# Unload
unload_plugin("my-plugin")
```

---

## C++ Availability Check

Check if C++ bindings are available:

```python
import pyflame as pf

if pf._CPP_AVAILABLE:
    # Full functionality available
    x = pf.randn([100, 100])
else:
    # Only Python ecosystem modules available
    # (tools, integrations, serving, benchmarks, extend)
    from pyflame.tools import Profiler
    from pyflame.integrations import WandbCallback
```
