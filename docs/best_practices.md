# PyFlame Best Practices

**PyFlame Version:** Pre-Release Alpha 1.0

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs described here are subject to change.

---

This guide covers best practices, optimization tips, and common pitfalls when developing with PyFlame.

---

## Table of Contents

1. [Understanding Lazy Evaluation](#understanding-lazy-evaluation)
2. [Memory Efficiency](#memory-efficiency)
3. [Graph Optimization](#graph-optimization)
4. [Layout Selection](#layout-selection)
5. [Numerical Stability](#numerical-stability)
6. [Type Best Practices](#type-best-practices)
7. [Common Pitfalls](#common-pitfalls)
8. [Performance Tips](#performance-tips)
9. [Debugging Techniques](#debugging-techniques)

---

## Understanding Lazy Evaluation

### How It Works

PyFlame builds a computation graph rather than executing operations immediately. This enables whole-graph optimization and efficient WSE code generation.

```python
import pyflame as pf

# These lines BUILD the graph - no computation happens
a = pf.randn([1000, 1000])
b = pf.randn([1000, 1000])
c = a @ b                    # Graph: matmul node created
d = pf.relu(c)               # Graph: relu node added
e = d.sum()                  # Graph: sum node added

# NOW computation happens
pf.eval(e)
```

### Best Practice: Batch Your Evaluations

**Do:** Evaluate once after building complete graph
```python
# Good: Single evaluation for entire computation
x = pf.randn([100, 100])
y = compute_layer1(x)
z = compute_layer2(y)
output = compute_layer3(z)
pf.eval(output)  # One evaluation
```

**Avoid:** Multiple evaluations in tight loops
```python
# Bad: Evaluation inside loop prevents graph optimization
for i in range(100):
    x = pf.randn([100])
    y = pf.relu(x)
    pf.eval(y)  # Evaluation each iteration - inefficient!
```

### When Evaluation Happens Implicitly

Be aware that these operations trigger evaluation:

```python
tensor.numpy()      # Convert to NumPy
print(tensor)       # Print values
tensor.data()       # Access raw data (C++)
```

---

## Memory Efficiency

### Reuse Tensors When Possible

Instead of creating new tensors repeatedly:

```python
# Good: Create once, use many times
weights = pf.randn([1000, 1000])
bias = pf.zeros([1000])

for batch in batches:
    output = batch @ weights + bias
    # Process output...
```

### Let Python Clean Up

PyFlame tensors are reference-counted. Allow scope to clean up:

```python
def process_batch(batch):
    # Local tensors freed when function returns
    temp1 = pf.relu(batch)
    temp2 = pf.sigmoid(temp1)
    result = temp2.sum()
    pf.eval(result)
    return result.numpy()  # Return NumPy, PyFlame tensors freed
```

### Avoid Holding References Unnecessarily

```python
# Bad: Keeping all intermediate results
results = []
for batch in batches:
    x = pf.from_numpy(batch)
    y = model.forward(x)
    results.append(y)  # Holds PyFlame tensors

# Good: Convert to NumPy immediately
results = []
for batch in batches:
    x = pf.from_numpy(batch)
    y = model.forward(x)
    pf.eval(y)
    results.append(y.numpy())  # Holds NumPy arrays
```

---

## Graph Optimization

### Keep Graphs Contiguous

PyFlame optimizes contiguous subgraphs better:

```python
# Good: Contiguous graph
x = pf.randn([100, 100])
y = pf.relu(x @ w1 + b1)
z = pf.relu(y @ w2 + b2)
output = pf.softmax(z, dim=1)
pf.eval(output)

# Suboptimal: Interrupted by NumPy
x = pf.randn([100, 100])
y = pf.relu(x @ w1 + b1)
pf.eval(y)
y_np = y.numpy()
y_np = custom_numpy_op(y_np)  # Breaks the graph
y2 = pf.from_numpy(y_np)
output = pf.softmax(y2 @ w2, dim=1)
pf.eval(output)
```

### Leverage Operation Fusion

The compiler can fuse certain operations. Structure code to enable this:

```python
# Fusable: Matmul + Bias + Activation
y = pf.relu(x @ w + b)

# Also fusable: Matmul + Bias + Activation + Reduction
loss = pf.relu(x @ w + b).sum()
```

---

## Layout Selection

### Match Layout to Access Pattern

**Row-major operations:** Use row partitioning
```python
# Matrix-vector multiply along rows
A = pf.randn([4096, 512], layout=pf.MeshLayout.row_partition(16))
x = pf.randn([512])
y = A @ x  # Each PE handles 256 rows
```

**Column-major operations:** Use column partitioning
```python
# Transpose then multiply
A = pf.randn([512, 4096], layout=pf.MeshLayout.col_partition(16))
x = pf.randn([512])
y = A.transpose(0, 1) @ x
```

**Large square matrices:** Use 2D grid
```python
# Large matrix-matrix multiply
A = pf.randn([4096, 4096], layout=pf.MeshLayout.grid(16, 16))
B = pf.randn([4096, 4096], layout=pf.MeshLayout.grid(16, 16))
C = A @ B  # Distributed across 256 PEs
```

### Start with SinglePE for Development

During development, use the default single-PE layout:

```python
# Development: Use default layout
x = pf.randn([1000, 1000])  # Implicit single_pe()

# Production: Specify distributed layout
x = pf.randn([1000, 1000], layout=pf.MeshLayout.grid(4, 4))
```

### Ensure Compatible Layouts

For operations between tensors, layouts should be compatible:

```python
# Good: Same layout
layout = pf.MeshLayout.row_partition(8)
a = pf.randn([1024, 512], layout=layout)
b = pf.randn([1024, 512], layout=layout)
c = a + b  # Element-wise, same distribution

# Careful: Different layouts may require data movement
layout1 = pf.MeshLayout.row_partition(8)
layout2 = pf.MeshLayout.col_partition(8)
a = pf.randn([512, 512], layout=layout1)
b = pf.randn([512, 512], layout=layout2)
c = a + b  # May require redistribution
```

---

## Numerical Stability

### Use Log-Space for Products

```python
# Bad: May overflow/underflow
product = pf.exp(log_values).prod()

# Good: Stay in log space
log_product = log_values.sum()
```

### Add Epsilon to Denominators

```python
# Bad: Division by zero possible
normalized = x / x.sum(dim=1, keepdim=True)

# Good: Add small epsilon
eps = 1e-8
normalized = x / (x.sum(dim=1, keepdim=True) + eps)
```

### Use Numerically Stable Functions

```python
# Use log_softmax instead of log(softmax())
# Bad: Potential numerical issues
probs = pf.softmax(logits, dim=1)
log_probs = pf.log(probs)

# Good: Numerically stable
log_probs = pf.log_softmax(logits, dim=1)
```

### Scale Attention Scores

```python
import numpy as np

# Standard scaled dot-product attention
d_k = query.shape[-1]
scale = 1.0 / np.sqrt(d_k)
scores = (query @ key.transpose(-2, -1)) * scale
attention = pf.softmax(scores, dim=-1)
```

---

## Type Best Practices

### Default to float32

```python
# float32 is the safest default
x = pf.randn([100, 100])  # dtype=float32 by default
x = pf.randn([100, 100], dtype=pf.float32)  # Explicit
```

### Use float16 for Memory-Constrained Scenarios

```python
# Half precision for large models
x = pf.randn([10000, 10000], dtype=pf.float16)
# Note: May have precision issues for small values
```

### Match NumPy Types

```python
import numpy as np

# Ensure type compatibility
data = your_numpy_array.astype(np.float32)
tensor = pf.from_numpy(data)

# Always specify float32 when converting
arr = np.ascontiguousarray(data, dtype=np.float32)
tensor = pf.from_numpy(arr)
```

---

## Common Pitfalls

### Pitfall 1: Shape Mismatch in MatMul

```python
# Wrong: Inner dimensions don't match
a = pf.randn([100, 50])
b = pf.randn([60, 75])
c = a @ b  # Error: 50 != 60

# Right: Inner dimensions match
a = pf.randn([100, 50])
b = pf.randn([50, 75])
c = a @ b  # Works: [100, 75]
```

### Pitfall 2: Forgetting to Evaluate

```python
# Wrong: Tensor not evaluated, numpy() may not reflect computation
x = pf.randn([100])
y = pf.relu(x)
result = y.numpy()  # Works, but triggers implicit eval

# Right: Explicit evaluation
x = pf.randn([100])
y = pf.relu(x)
pf.eval(y)  # Explicit
result = y.numpy()
```

### Pitfall 3: Using Python Loops Instead of Tensor Operations

```python
# Wrong: Slow Python loop
result = pf.zeros([100])
for i in range(100):
    result[i] = x[i] + y[i]  # If indexing were supported

# Right: Vectorized operation
result = x + y
```

### Pitfall 4: Creating Tensors in Tight Loops

```python
# Wrong: Creating weights each iteration
for batch in batches:
    w = pf.randn([100, 100])  # New tensor each time!
    output = batch @ w

# Right: Create once, reuse
w = pf.randn([100, 100])
for batch in batches:
    output = batch @ w
```

### Pitfall 5: Not Checking Shapes

```python
# Good practice: Verify shapes
print(f"Input shape: {x.shape}")
print(f"Weight shape: {w.shape}")

y = x @ w
print(f"Output shape: {y.shape}")
assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
```

---

## Performance Tips

### Tip 1: Prefer Larger Batch Sizes

```python
# Better throughput with larger batches
batch_size = 256  # or larger if memory permits
x = pf.randn([batch_size, features])
```

### Tip 2: Minimize Data Transfers

```python
# Bad: Multiple transfers
for i in range(n):
    x = pf.from_numpy(data[i])
    y = model(x)
    pf.eval(y)
    results.append(y.numpy())

# Better: Batch transfers
x = pf.from_numpy(all_data)  # Single transfer in
y = model(x)
pf.eval(y)
results = y.numpy()  # Single transfer out
```

### Tip 3: Use Contiguous Memory

```python
import numpy as np

# Ensure contiguous memory layout
arr = np.ascontiguousarray(your_array)
tensor = pf.from_numpy(arr)
```

### Tip 4: Profile Your Code

```python
import time

# Simple profiling
start = time.perf_counter()

x = pf.randn([1000, 1000])
y = pf.randn([1000, 1000])
z = x @ y
pf.eval(z)

elapsed = time.perf_counter() - start
print(f"Elapsed: {elapsed:.4f}s")
```

### Tip 5: Enable Optimizations in CodeGen

```python
options = pf.CodeGenOptions()
options.optimize = True  # Enable graph optimizations
options.generate_debug_info = False  # Disable for production
```

---

## Debugging Techniques

### Inspect the Computation Graph

```python
# Print the graph structure
x = pf.randn([100, 100])
y = pf.relu(x @ w + b)
z = y.sum()

pf.print_graph(z)
```

### Check Tensor Properties

```python
x = pf.randn([100, 100])
print(f"Shape: {x.shape}")
print(f"DType: {x.dtype}")
print(f"Evaluated: {x.is_evaluated()}")
print(f"Lazy: {pf.is_lazy(x)}")
```

### Verify Intermediate Values

```python
# Check values at intermediate points
x = pf.randn([100])
y = pf.relu(x)
pf.eval(y)

# Check statistics
arr = y.numpy()
print(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}")
print(f"Has NaN: {np.isnan(arr).any()}")
print(f"Has Inf: {np.isinf(arr).any()}")
```

### Compare with NumPy Reference

```python
import numpy as np

# Create identical input
np.random.seed(42)
data = np.random.randn(100, 100).astype(np.float32)

# NumPy reference
np_result = np.maximum(0, data @ weights)

# PyFlame
x = pf.from_numpy(data)
w = pf.from_numpy(weights)
pf_result = pf.relu(x @ w)
pf.eval(pf_result)

# Compare
diff = np.abs(np_result - pf_result.numpy())
print(f"Max difference: {diff.max()}")
np.testing.assert_allclose(np_result, pf_result.numpy(), rtol=1e-5)
```

### Use Debug Code Generation

```python
options = pf.CodeGenOptions()
options.generate_debug_info = True
options.emit_comments = True

result = pf.compile_to_csl(tensor, options)
# Inspect generated code for debugging
```

---

## Summary Checklist

Before deploying your PyFlame code:

- [ ] Evaluate tensors in batches, not inside tight loops
- [ ] Use appropriate data types (float32 by default)
- [ ] Verify tensor shapes match expected dimensions
- [ ] Add epsilon to prevent division by zero
- [ ] Use numerically stable functions (log_softmax, etc.)
- [ ] Choose layouts appropriate for your access patterns
- [ ] Minimize data transfers between NumPy and PyFlame
- [ ] Profile critical sections for performance
- [ ] Compare results with NumPy reference implementation
- [ ] Enable optimizations for production builds

---

## See Also

- [Getting Started](getting_started.md) - Introduction tutorial
- [API Reference](api_reference.md) - Complete function documentation
- [Examples](examples.md) - Practical code examples
- [Integration Guide](integration_guide.md) - Adding PyFlame to your project
