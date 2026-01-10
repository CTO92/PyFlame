# Getting Started with PyFlame

**PyFlame Version:** Pre-Release Alpha 1.0

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs described here are subject to change.

---

## Introduction

PyFlame is a tensor computation library designed for the Cerebras Wafer-Scale Engine (WSE). This guide will help you get started with PyFlame, from installation to running your first computation.

### What You'll Learn

1. Installing PyFlame
2. Creating your first tensors
3. Building computation graphs
4. Executing computations
5. Understanding lazy evaluation

---

## 1. Installation

### Prerequisites

- **Python 3.8+**
- **C++17 compiler** (GCC 9+, Clang 10+, or MSVC 2019+)
- **CMake 3.18+**

### Building from Source

```bash
# Clone the repository
git clone https://github.com/CTO92/PyFlame.git
cd PyFlame

# Create build directory
mkdir build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Run tests to verify installation
ctest --output-on-failure
```

### Python Installation

After building, install the Python package:

```bash
# From the repository root
pip install -e .
```

Verify the installation:

```python
import pyflame as pf
print(f"PyFlame version: {pf.__version__}")
print(f"Release status: {pf.__release_status__}")
```

---

## 2. Your First Tensor

### Creating Tensors

PyFlame provides several ways to create tensors:

```python
import pyflame as pf

# Create a tensor filled with zeros
a = pf.zeros([3, 4])
print(f"Shape: {a.shape}, dtype: {a.dtype}")

# Create a tensor filled with ones
b = pf.ones([3, 4])

# Create a tensor with random values (normal distribution)
c = pf.randn([3, 4])

# Create a tensor with random values (uniform [0, 1))
d = pf.rand([3, 4])

# Create a tensor with a specific value
e = pf.full([3, 4], 3.14)

# Create a range of values
f = pf.arange(0, 10)  # [0, 1, 2, ..., 9]
```

### Data Types

PyFlame supports several data types:

```python
# Specify data type when creating tensors
x = pf.zeros([100, 100], dtype=pf.float32)  # Default
y = pf.zeros([100, 100], dtype=pf.float16)  # Half precision
z = pf.zeros([100, 100], dtype=pf.int32)    # Integer
```

Available types: `float32`, `float16`, `bfloat16`, `int32`, `int16`, `int8`, `bool_`

### From NumPy

```python
import numpy as np

# Convert from NumPy
np_array = np.random.randn(3, 4).astype(np.float32)
tensor = pf.from_numpy(np_array)

# Or use the tensor() function
tensor = pf.tensor([[1, 2, 3], [4, 5, 6]])
```

---

## 3. Tensor Operations

### Arithmetic Operations

```python
a = pf.randn([100, 100])
b = pf.randn([100, 100])

# Basic arithmetic
c = a + b    # Addition
d = a - b    # Subtraction
e = a * b    # Element-wise multiplication
f = a / b    # Element-wise division

# Scalar operations
g = a + 5.0
h = a * 2.0
```

### Matrix Operations

```python
# Matrix multiplication
a = pf.randn([100, 50])
b = pf.randn([50, 75])

c = a @ b              # Using @ operator
c = pf.matmul(a, b)    # Using function
```

### Activation Functions

```python
x = pf.randn([100, 100])

# Available activations
y = pf.relu(x)
y = pf.sigmoid(x)
y = pf.tanh(x)
y = pf.gelu(x)
y = pf.silu(x)
y = pf.softmax(x, dim=1)
```

### Reductions

```python
x = pf.randn([100, 100])

# Reduce operations
total = x.sum()           # Sum all elements
mean = x.mean()           # Mean of all elements
maximum = x.max()         # Maximum value
minimum = x.min()         # Minimum value

# Reduce along specific dimension
row_sums = x.sum(dim=1)           # Sum each row
col_means = x.mean(dim=0)         # Mean of each column
row_max = x.max(dim=1, keepdim=True)  # Keep dimensions
```

### Math Functions

```python
x = pf.randn([100, 100])

y = pf.abs(x)    # Absolute value
y = pf.sqrt(x)   # Square root (for positive values)
y = pf.exp(x)    # Exponential
y = pf.log(x)    # Natural logarithm (for positive values)
y = pf.sin(x)    # Sine
y = pf.cos(x)    # Cosine
```

---

## 4. Understanding Lazy Evaluation

### Key Concept

PyFlame uses **lazy evaluation** - operations don't execute immediately. Instead, they build a computation graph that is executed when you explicitly request results.

```python
import pyflame as pf

# These lines build the graph - NO computation happens yet
a = pf.randn([1000, 1000])
b = pf.randn([1000, 1000])
c = a @ b
d = pf.relu(c)
e = d.sum()

# Check if evaluated
print(pf.is_lazy(e))  # True - not yet computed

# NOW computation happens
result = pf.eval(e)

print(pf.is_lazy(e))  # False - now computed
print(result.numpy()) # Get the actual value
```

### Why Lazy Evaluation?

1. **Optimization**: The entire graph is visible for optimization
2. **Batching**: Multiple operations are compiled together
3. **WSE Compatibility**: The WSE requires static graphs for compilation

### Triggering Evaluation

Computation is triggered when you:

```python
# Explicit evaluation
result = pf.eval(tensor)
tensor.eval()

# Implicit evaluation (also triggers computation)
value = tensor.numpy()     # Convert to NumPy
print(tensor)              # Print values
```

---

## 5. A Complete Example

Here's a complete example of building a simple computation:

```python
import pyflame as pf

def simple_mlp_forward(x, weights, biases):
    """Simple 2-layer MLP forward pass."""
    # Layer 1
    h = x @ weights[0] + biases[0]
    h = pf.relu(h)

    # Layer 2
    out = h @ weights[1] + biases[1]
    return out

# Create input and parameters
batch_size = 32
input_dim = 784
hidden_dim = 256
output_dim = 10

x = pf.randn([batch_size, input_dim])
w1 = pf.randn([input_dim, hidden_dim]) * 0.01
b1 = pf.zeros([hidden_dim])
w2 = pf.randn([hidden_dim, output_dim]) * 0.01
b2 = pf.zeros([output_dim])

# Forward pass (builds graph, doesn't compute yet)
logits = simple_mlp_forward(x, [w1, w2], [b1, b2])
probs = pf.softmax(logits, dim=1)

# Now evaluate
pf.eval(probs)

# Get results
print(f"Output shape: {probs.shape}")
print(f"First sample probabilities: {probs.numpy()[0]}")
```

---

## 6. Working with Layouts (Advanced)

For Cerebras WSE execution, you can specify how tensors are distributed across Processing Elements (PEs):

```python
import pyflame as pf

# Single PE (default)
a = pf.zeros([100, 100], layout=pf.MeshLayout.single_pe())

# Distribute rows across 4 PEs
b = pf.zeros([100, 100], layout=pf.MeshLayout.row_partition(4))

# Distribute columns across 4 PEs
c = pf.zeros([100, 100], layout=pf.MeshLayout.col_partition(4))

# 2D grid distribution (4x4 = 16 PEs)
d = pf.zeros([100, 100], layout=pf.MeshLayout.grid(4, 4))
```

> **Note:** Layout specifications are used for WSE code generation. The CPU reference implementation ignores layouts and executes all operations on a single thread.

---

## 7. Inspecting Computation Graphs

For debugging, you can inspect the computation graph:

```python
import pyflame as pf

a = pf.randn([100, 100])
b = pf.randn([100, 100])
c = a @ b
d = pf.relu(c)
e = d.sum()

# Print the computation graph
pf.print_graph(e)

# Get graph object for inspection
graph = pf.get_graph(e)
```

---

## Next Steps

- Read the [API Reference](api_reference.md) for complete documentation
- See [Examples](examples.md) for more complex use cases
- Check [Best Practices](best_practices.md) for optimization tips
- Review [Integration Guide](integration_guide.md) to add PyFlame to your project

---

## Getting Help

- **GitHub Issues**: https://github.com/CTO92/PyFlame/issues
- **Documentation**: See the `docs/` directory for design documents
