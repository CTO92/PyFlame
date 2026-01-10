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

### Utilities
19. [Utility Functions](#utility-functions)

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
