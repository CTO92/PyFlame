# PyTorch to PyFlame Migration Guide

**PyFlame Version:** Pre-Release Alpha 1.0

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs described here are subject to change.

---

## Overview

PyFlame is a native deep learning framework designed specifically for Cerebras Wafer-Scale Engine (WSE) hardware. While PyFlame provides a familiar API for PyTorch developers, the underlying execution model and hardware architecture are fundamentally different. This guide explains the key differences and how to adapt your software design patterns when migrating from PyTorch to PyFlame.

---

## Table of Contents

1. [Fundamental Architecture Differences](#fundamental-architecture-differences)
2. [Execution Model: Eager vs Lazy](#execution-model-eager-vs-lazy)
3. [Hardware Topology: GPU vs 2D PE Mesh](#hardware-topology-gpu-vs-2d-pe-mesh)
4. [Graph Compilation: Dynamic vs Static](#graph-compilation-dynamic-vs-static)
5. [Model Construction Patterns](#model-construction-patterns)
6. [Data Flow and Control Flow](#data-flow-and-control-flow)
7. [Memory Model Differences](#memory-model-differences)
8. [Common Migration Patterns](#common-migration-patterns)
9. [What Works the Same](#what-works-the-same)
10. [What Requires Rethinking](#what-requires-rethinking)

---

## Fundamental Architecture Differences

### Conceptual Comparison

| Aspect | PyTorch (CUDA/GPU) | PyFlame (Cerebras WSE) |
|--------|-------------------|------------------------|
| **Execution Model** | Eager (immediate) | Lazy (deferred) |
| **Hardware** | GPU (SIMD cores) | 2D mesh of PEs |
| **Memory** | Global GPU memory | Distributed per-PE SRAM |
| **Parallelism** | Data parallelism, tensor cores | Spatial dataflow |
| **Communication** | PCIe/NVLink | Wavelet routing |
| **Graph Model** | Dynamic (define-by-run) | Static (compile-once) |
| **Recompilation** | Every forward pass | Single compile, many runs |

### Design Philosophy

**PyTorch:** "Make it easy to experiment"
- Dynamic computation graphs
- Immediate feedback
- Python-native debugging
- Flexibility over optimization

**PyFlame:** "Maximize WSE hardware utilization"
- Static graphs for whole-program optimization
- Compile-time PE placement
- Native 2D mesh awareness
- Determinism over flexibility

---

## Execution Model: Eager vs Lazy

### PyTorch: Eager Execution

In PyTorch, operations execute immediately when called:

```python
import torch

x = torch.randn(100, 100)      # Memory allocated, random values computed NOW
y = torch.relu(x)               # ReLU computed NOW
z = y.sum()                     # Sum computed NOW
print(z)                        # Value available immediately
```

### PyFlame: Lazy Evaluation

In PyFlame, operations build a computation graph that executes later:

```python
import pyflame as pf

x = pf.randn([100, 100])       # Graph node created - NO computation yet
y = pf.relu(x)                  # Graph extended - still NO computation
z = y.sum()                     # Graph extended - still NO computation

# NOW computation happens (entire graph compiled and executed)
pf.eval(z)                      # or z.numpy() triggers implicit eval
print(z.numpy())
```

### Why This Matters

**PyTorch approach:**
- Debug by printing intermediate values anytime
- Each operation can fail independently
- Memory used incrementally as operations execute
- Easy to mix with Python control flow

**PyFlame approach:**
- Must explicitly trigger evaluation
- Entire graph succeeds or fails together
- Memory planning happens for whole graph
- Enables whole-graph optimization

### Adapting Your Code

```python
# PyTorch debugging pattern - WON'T WORK in PyFlame
def debug_forward(x):
    h1 = layer1(x)
    print(h1.mean())  # In PyTorch: prints value. In PyFlame: triggers unexpected eval!
    h2 = layer2(h1)
    return h2

# PyFlame-friendly debugging pattern
def debug_forward(x):
    h1 = layer1(x)
    h2 = layer2(h1)
    return h1, h2  # Return intermediates

# Debug after building graph
h1, h2 = debug_forward(x)
pf.eval(h1, h2)  # Evaluate both
print(h1.numpy().mean())  # Now inspect
```

---

## Hardware Topology: GPU vs 2D PE Mesh

### GPU Architecture (PyTorch)

GPUs have:
- Global memory accessible by all cores
- Streaming multiprocessors (SMs) with shared memory
- All threads can access any memory location
- CUDA abstracts physical layout from programmer

```python
# PyTorch: Layout is abstracted
x = torch.randn(1000, 1000, device='cuda')  # "Somewhere" on GPU
y = torch.randn(1000, 1000, device='cuda')  # "Somewhere" on GPU
z = x @ y  # CUDA handles parallelization
```

### Cerebras WSE Architecture (PyFlame)

The WSE has:
- 850,000+ independent Processing Elements (PEs)
- Each PE has its own 48KB SRAM
- 2D mesh interconnect between PEs
- Data must be explicitly distributed

```python
# PyFlame: Layout is explicit and matters!
layout = pf.MeshLayout.grid(16, 16)  # Use a 16x16 grid of PEs

x = pf.randn([4096, 4096], layout=layout)  # Distributed across 256 PEs
y = pf.randn([4096, 4096], layout=layout)  # Same distribution
z = x @ y  # Compiler generates PE-to-PE communication
```

### Key Differences

| GPU | WSE |
|-----|-----|
| All threads see same memory | Each PE has private memory |
| Implicit data movement | Explicit data distribution |
| Memory bandwidth limited | Wavelet bandwidth 20 PB/s |
| Global synchronization | Local neighbor communication |

### Implications for Software Design

**PyTorch patterns that don't translate well:**
```python
# Random access patterns - expensive on WSE
output[indices] = values  # Gather/scatter

# Global reductions without planning
global_mean = tensor.mean()  # Requires all-to-one communication
```

**PyFlame-optimized patterns:**
```python
# Tile operations to match PE topology
layout = pf.MeshLayout.grid(8, 8)
x = pf.randn([2048, 2048], layout=layout)  # 256x256 per PE

# Local reductions then global
local_sum = x.sum(dim=1)  # Each PE reduces its portion
global_sum = local_sum.sum()  # Collect from all PEs
```

---

## Graph Compilation: Dynamic vs Static

### PyTorch: Dynamic Graphs

PyTorch rebuilds the computation graph every forward pass:

```python
# This works fine in PyTorch
def forward(self, x, use_dropout=True):
    h = self.linear1(x)
    if use_dropout:  # Different graph each call!
        h = F.dropout(h, p=0.5, training=self.training)
    return self.linear2(h)
```

### PyFlame: Static Graphs

PyFlame compiles the graph once. The graph structure cannot change between executions:

```python
# This will NOT work as expected in PyFlame
def forward(self, x, use_dropout=True):
    h = self.linear1(x)
    if use_dropout:  # Python if evaluates at GRAPH BUILD time!
        h = pf.dropout(h, p=0.5)  # Either always in graph or never
    return self.linear2(h)
```

### Why Static Graphs?

The Cerebras WSE compiles the entire computation graph to a hardware configuration:
- PE placement is fixed at compile time
- Wavelet routing is pre-computed
- Memory allocation is planned in advance
- **Recompilation is not possible during execution**

### Adapting Dynamic Patterns

**Pattern 1: Training vs Inference**

```python
# PyTorch: Mode checked at runtime
class Model(nn.Module):
    def forward(self, x):
        if self.training:  # Dynamic check
            x = F.dropout(x, 0.5)
        return self.linear(x)

# PyFlame: Compile separate graphs
class Model(pf.nn.Module):
    def forward_train(self, x):
        x = pf.dropout(x, 0.5)
        return self.linear(x)

    def forward_eval(self, x):
        return self.linear(x)

# Use different methods
train_output = model.forward_train(x)
eval_output = model.forward_eval(x)
```

**Pattern 2: Variable Sequence Lengths**

```python
# PyTorch: Different length each batch
for batch in dataloader:
    # batch.shape = [batch_size, varying_seq_len, features]
    output = model(batch)  # Works with any length

# PyFlame: Pad to fixed length
MAX_SEQ_LEN = 512
for batch in dataloader:
    padded = pad_to_length(batch, MAX_SEQ_LEN)
    mask = create_padding_mask(batch, MAX_SEQ_LEN)
    output = model(padded, mask)  # Fixed shape, use mask
```

**Pattern 3: Conditional Computation**

```python
# PyTorch: Python conditionals work
def forward(self, x, task_type):
    if task_type == "classification":
        return self.classifier(x)
    else:
        return self.regressor(x)

# PyFlame: Compute all, select with tensor ops
def forward(self, x, task_one_hot):
    # task_one_hot: [batch, 2] where [:, 0] = classification, [:, 1] = regression
    class_out = self.classifier(x)   # Always computed
    reg_out = self.regressor(x)       # Always computed

    # Select using tensor operations (compiled into graph)
    output = class_out * task_one_hot[:, 0:1] + reg_out * task_one_hot[:, 1:2]
    return output
```

---

## Model Construction Patterns

### PyTorch Model Pattern

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)  # Skip connection
        return x
```

### PyFlame Model Pattern

```python
import pyflame as pf
from pyflame import nn

class ResBlock(nn.Module):
    def __init__(self, channels, layout=None):
        super().__init__()
        # Layout awareness: specify how weights are distributed
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, layout=layout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, layout=layout)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layout = layout

    def forward(self, x):
        residual = x
        x = pf.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = pf.relu(x + residual)
        return x
```

### Key Differences in Model Construction

| Aspect | PyTorch | PyFlame |
|--------|---------|---------|
| Weight initialization | Implicit in layer creation | May need layout specification |
| Forward pass | Executes immediately | Builds graph |
| BatchNorm | Tracks running stats dynamically | Stats must be handled carefully |
| Dropout | Different each call | Fixed pattern or use mask |
| Weight sharing | Reference same module | Same, but layout must match |

### Handling State Across Steps

**PyTorch:** Optimizer and BatchNorm maintain running state automatically

```python
# Works automatically
for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()  # Updates running averages, momentum, etc.
```

**PyFlame:** State management is more explicit

```python
# State must be managed carefully
trainer = pf.training.Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    config=pf.training.TrainerConfig(
        max_epochs=epochs,
        # State tracking configured here
    )
)
trainer.fit(train_loader)
```

---

## Data Flow and Control Flow

### PyTorch: Python Control Flow

```python
# PyTorch: Python loops work normally
def attention(query, key, value, mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1))

    if mask is not None:  # Python if works
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    # Dynamic loop
    for _ in range(self.num_refinements):
        weights = self.refine(weights)

    return torch.matmul(weights, value)
```

### PyFlame: Tensor-Based Control Flow

```python
# PyFlame: Use tensor operations for "control flow"
def attention(query, key, value, mask=None):
    scores = pf.matmul(query, key.transpose(-2, -1))

    # Use tensor operations instead of Python if
    if mask is not None:
        # Mask is a tensor, not Python conditional
        scores = pf.where(mask == 0, pf.full_like(scores, float('-inf')), scores)

    weights = pf.softmax(scores, dim=-1)

    # Unroll loops at graph construction time
    # OR use fixed iteration count
    for _ in range(self.num_refinements):  # Must be fixed at graph build
        weights = self.refine(weights)

    return pf.matmul(weights, value)
```

### Loop Patterns

**PyTorch dynamic loops:**
```python
# Variable iteration count
while not converged:
    x = update(x)
    converged = check_convergence(x)
```

**PyFlame static loops:**
```python
# Fixed iteration count (unrolled at compile time)
MAX_ITERATIONS = 100
for i in range(MAX_ITERATIONS):
    x = update(x)
    # Can't break early, but can use damping:
    # x = converged_mask * x + (1 - converged_mask) * update(x)
```

### Branching Patterns

**PyTorch:**
```python
if condition:
    result = path_a(x)
else:
    result = path_b(x)
```

**PyFlame:**
```python
# Both paths computed, select result
result_a = path_a(x)
result_b = path_b(x)
# condition must be a tensor
result = pf.where(condition, result_a, result_b)
```

---

## Memory Model Differences

### PyTorch Memory Model

- Single large GPU memory pool
- Allocate/free dynamically during execution
- Memory defragmentation at runtime
- OOM errors happen during forward/backward pass

```python
# PyTorch: Can allocate any time
intermediate = torch.zeros(huge_size)  # Allocates NOW
# ... use intermediate ...
del intermediate  # Frees NOW
# Memory available for reuse immediately
```

### PyFlame Memory Model

- Memory distributed across PEs (48KB per PE)
- All memory planned at compile time
- No dynamic allocation during execution
- Memory errors happen at compile time

```python
# PyFlame: Memory planned ahead
intermediate = pf.zeros([huge_size])  # Node in graph, no allocation yet
pf.eval(intermediate)  # Compiler plans all memory, THEN allocates

# Memory layout affects performance dramatically
# Small tensor: single PE
small = pf.randn([100])  # Fits in one PE's SRAM

# Large tensor: distributed
large = pf.randn([1000000], layout=pf.MeshLayout.row_partition(1000))
# Split across 1000 PEs
```

### Implications

**PyTorch patterns that may not work:**
```python
# Growing lists of tensors
history = []
for step in range(1000):
    output = model(x)
    history.append(output)  # Accumulates memory
```

**PyFlame-friendly patterns:**
```python
# Pre-allocate or stream to CPU
history_np = []
for step in range(1000):
    output = model(x)
    pf.eval(output)
    history_np.append(output.numpy())  # Move to CPU, free WSE memory
```

---

## Common Migration Patterns

### Pattern 1: Simple Forward Pass

**PyTorch:**
```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
output = model(input_tensor)  # Immediate execution
```

**PyFlame:**
```python
class SimpleNet(pf.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = pf.nn.Linear(784, 256)
        self.fc2 = pf.nn.Linear(256, 10)

    def forward(self, x):
        x = pf.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
output = model(input_tensor)  # Builds graph
pf.eval(output)               # Executes graph
```

### Pattern 2: Training Loop

**PyTorch:**
```python
for epoch in range(epochs):
    for batch, target in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")  # Immediate value
```

**PyFlame:**
```python
# Option 1: Use Trainer (recommended)
trainer = pf.training.Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=criterion
)
trainer.fit(train_loader)

# Option 2: Manual loop
for epoch in range(epochs):
    for batch, target in train_loader:
        batch = pf.from_numpy(batch)
        target = pf.from_numpy(target)

        with pf.autograd.enable_grad():
            output = model(batch)
            loss = criterion(output, target)

        pf.backward(loss)
        optimizer.step()
        pf.eval(loss)  # Must eval before accessing value
        print(f"Loss: {loss.numpy()}")
```

### Pattern 3: Inference with Batching

**PyTorch:**
```python
model.eval()
with torch.no_grad():
    results = []
    for batch in dataloader:
        output = model(batch)
        results.append(output.cpu().numpy())
```

**PyFlame:**
```python
model.eval()
engine = pf.serving.InferenceEngine(model)
engine.warmup(example_input)

results = []
for batch in dataloader:
    batch = pf.from_numpy(batch)
    output = engine.infer(batch)
    results.append(output.numpy())
```

### Pattern 4: Custom Autograd Function

**PyTorch:**
```python
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
```

**PyFlame:**
```python
class CustomReLU(pf.extend.AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return pf.maximum(input, 0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask = input > 0
        return grad_output * mask
```

---

## What Works the Same

These patterns translate directly from PyTorch to PyFlame:

### Tensor Operations
```python
# Basic arithmetic (same API)
z = x + y
z = x * y
z = x @ y  # Matrix multiply

# Reductions (same API)
mean = x.mean()
total = x.sum(dim=1)

# Activations (same API)
y = pf.relu(x)
y = pf.sigmoid(x)
y = pf.softmax(x, dim=-1)
```

### Layer Definitions
```python
# Module pattern (same structure)
class MyLayer(pf.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = pf.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
```

### Shape Operations
```python
# Reshape, transpose work similarly
x = x.view([batch, -1])
x = x.transpose(0, 1)
x = x.reshape([new_shape])
```

---

## What Requires Rethinking

### 1. Dynamic Shapes

**Problem:** PyFlame requires fixed shapes at compile time.

**Solution:** Use fixed dimensions with padding/masking.

```python
# Instead of variable sequence length
# Use: fixed max length + attention mask
MAX_LEN = 512
padded_input = pad_sequence(input, MAX_LEN)
mask = create_mask(original_lengths, MAX_LEN)
output = model(padded_input, mask)
```

### 2. Conditional Execution

**Problem:** Python `if` statements don't create conditional computation.

**Solution:** Use tensor operations for all data-dependent branching.

```python
# Instead of: if x > threshold: result = a else: result = b
# Use:
result = pf.where(x > threshold, a, b)
```

### 3. Dynamic Module Selection

**Problem:** Can't dynamically choose which modules to execute.

**Solution:** Execute all paths, combine with selection tensor.

```python
# Instead of: expert = experts[routing_index]
# Use:
outputs = [expert(x) for expert in experts]  # Run all
stacked = pf.stack(outputs, dim=0)
result = (stacked * routing_weights.unsqueeze(-1)).sum(dim=0)
```

### 4. In-Place Operations

**Problem:** In-place ops can complicate graph building.

**Solution:** Prefer functional style; let compiler optimize.

```python
# Avoid: x.add_(y)
# Prefer: x = x + y
```

### 5. Python-Level Iteration During Forward

**Problem:** Python loops at forward time create graph structure.

**Solution:** Unroll loops or use fixed iteration counts.

```python
# For RNNs: unroll to max sequence length
for t in range(MAX_SEQ_LEN):
    hidden = rnn_cell(x[:, t], hidden)
```

### 6. Random Operations

**Problem:** Each `randn()` call creates a fixed pattern in the graph.

**Solution:** Use seeds or pre-computed random tensors.

```python
# For dropout: use pre-computed mask or seed-based RNG
dropout_mask = pf.bernoulli(pf.ones_like(x), p=keep_prob, seed=step)
x = x * dropout_mask / keep_prob
```

---

## Architecture Decision Checklist

When migrating a model, ask these questions:

- [ ] **Fixed shapes:** Are all tensor shapes known at compile time?
- [ ] **Static graph:** Is the computation graph the same for every input?
- [ ] **No dynamic branching:** Are all conditionals tensor-based?
- [ ] **Fixed loops:** Do all loops have fixed iteration counts?
- [ ] **Layout planning:** Have you considered PE distribution for large tensors?
- [ ] **Explicit evaluation:** Have you added `pf.eval()` calls?
- [ ] **State management:** Is optimizer/BatchNorm state handled properly?
- [ ] **Memory footprint:** Will intermediate tensors fit in PE SRAM?

---

## Summary

| If in PyTorch you... | In PyFlame you should... |
|---------------------|--------------------------|
| Call operations directly | Build graph, then `pf.eval()` |
| Use Python `if` for data-dependent logic | Use `pf.where()` or tensor masks |
| Use variable-length sequences | Pad to fixed length, use masks |
| Use dynamic loops (`while`) | Use fixed `for` loops |
| Let PyTorch handle data placement | Specify `MeshLayout` explicitly |
| Use `model.train()`/`model.eval()` | Consider separate forward methods |
| Debug with `print(tensor)` | Eval first, then print |
| Rely on dynamic memory allocation | Plan memory at compile time |

---

## See Also

- [Getting Started](getting_started.md) - Introduction to PyFlame
- [Best Practices](best_practices.md) - Optimization tips
- [API Reference](api_reference.md) - Complete API documentation
- [Examples](examples.md) - Code examples for common patterns
