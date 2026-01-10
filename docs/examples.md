# PyFlame Examples

**PyFlame Version:** Pre-Release Alpha 1.0

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs described here are subject to change.

---

This document provides practical examples of using PyFlame for common deep learning tasks.

---

## Table of Contents

1. [Basic Tensor Operations](#basic-tensor-operations)
2. [Simple Neural Network](#simple-neural-network)
3. [Multi-Layer Perceptron](#multi-layer-perceptron)
4. [Batch Processing](#batch-processing)
5. [Attention Mechanism](#attention-mechanism)
6. [Distributed Computation with Layouts](#distributed-computation-with-layouts)
7. [CSL Code Generation](#csl-code-generation)
8. [Working with NumPy](#working-with-numpy)

---

## Basic Tensor Operations

### Vector Operations

```python
import pyflame as pf

# Create vectors
a = pf.randn([1000])
b = pf.randn([1000])

# Basic operations
c = a + b          # Element-wise addition
d = a * b          # Element-wise multiplication
e = a - b          # Subtraction
f = a / (b + 1e-8) # Division (add epsilon for stability)

# Scalar operations
g = a * 2.0
h = a + 5.0

# Evaluate everything at once
pf.eval(c, d, e, f, g, h)

# Check results
print(f"Sum result mean: {c.numpy().mean():.4f}")
print(f"Product result mean: {d.numpy().mean():.4f}")
```

### Matrix Operations

```python
import pyflame as pf

# Create matrices
A = pf.randn([100, 50])
B = pf.randn([50, 75])
C = pf.randn([100, 75])

# Matrix multiplication
D = A @ B  # [100, 75]

# Element-wise operations with matrices
E = D + C  # Broadcasting works automatically
F = D * C

# Transpose and multiply
G = A @ A.transpose(0, 1)  # [100, 100] - creates symmetric matrix

pf.eval(D, E, F, G)

print(f"Matrix multiply shape: {D.shape}")
print(f"Symmetric matrix diagonal mean: {G.numpy().diagonal().mean():.4f}")
```

### Reductions

```python
import pyflame as pf

# Create a batch of data
x = pf.randn([32, 100])  # 32 samples, 100 features

# Global reductions
total = x.sum()
average = x.mean()
maximum = x.max()
minimum = x.min()

# Per-sample reductions
sample_sums = x.sum(dim=1)      # [32]
sample_means = x.mean(dim=1)    # [32]
sample_maxes = x.max(dim=1)     # [32]

# Per-feature reductions
feature_means = x.mean(dim=0)   # [100]
feature_stds = ((x - feature_means) ** 2).mean(dim=0).sqrt()  # Manual std

pf.eval(sample_means, feature_means, feature_stds)

print(f"Sample means shape: {sample_means.shape}")
print(f"Feature means shape: {feature_means.shape}")
```

---

## Simple Neural Network

### Single Layer Network

```python
import pyflame as pf
import numpy as np

# Network parameters
input_dim = 784    # e.g., flattened 28x28 image
output_dim = 10    # e.g., 10 classes

# Initialize weights (Xavier initialization)
scale = np.sqrt(2.0 / input_dim)
weights = pf.randn([input_dim, output_dim]) * scale
bias = pf.zeros([output_dim])

# Forward pass function
def forward(x):
    logits = x @ weights + bias
    probs = pf.softmax(logits, dim=1)
    return probs

# Create input batch
batch_size = 32
x = pf.randn([batch_size, input_dim])

# Run forward pass
probs = forward(x)
pf.eval(probs)

print(f"Output shape: {probs.shape}")
print(f"Probabilities sum (should be 1): {probs.numpy()[0].sum():.6f}")
```

---

## Multi-Layer Perceptron

### Three-Layer MLP

```python
import pyflame as pf
import numpy as np

class MLP:
    """Simple Multi-Layer Perceptron."""

    def __init__(self, layer_sizes):
        """Initialize MLP with given layer sizes.

        Args:
            layer_sizes: List of layer dimensions, e.g., [784, 256, 128, 10]
        """
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]

            # Xavier initialization
            scale = np.sqrt(2.0 / in_dim)
            w = pf.randn([in_dim, out_dim]) * scale
            b = pf.zeros([out_dim])

            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        """Forward pass through the network."""
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            # Apply ReLU for all but last layer
            if i < len(self.weights) - 1:
                x = pf.relu(x)
        return x

    def predict(self, x):
        """Get class probabilities."""
        logits = self.forward(x)
        return pf.softmax(logits, dim=1)


# Create network
model = MLP([784, 256, 128, 10])

# Create batch
batch_size = 64
x = pf.randn([batch_size, 784])

# Forward pass
probs = model.predict(x)
pf.eval(probs)

# Get predictions
predictions = probs.numpy().argmax(axis=1)
print(f"Predictions: {predictions[:10]}")
```

---

## Batch Processing

### Processing Data in Batches

```python
import pyflame as pf
import numpy as np

def process_in_batches(data, batch_size, model_fn):
    """Process data in batches.

    Args:
        data: NumPy array of shape [N, features]
        batch_size: Number of samples per batch
        model_fn: Function that takes PyFlame tensor and returns output

    Returns:
        NumPy array of all outputs concatenated
    """
    n_samples = data.shape[0]
    outputs = []

    for i in range(0, n_samples, batch_size):
        # Get batch
        batch = data[i:i+batch_size]

        # Convert to PyFlame
        x = pf.from_numpy(batch.astype(np.float32))

        # Process
        y = model_fn(x)
        pf.eval(y)

        # Collect result
        outputs.append(y.numpy())

    return np.concatenate(outputs, axis=0)


# Example usage
def simple_model(x):
    w = pf.randn([100, 50])
    return pf.relu(x @ w)

# Generate data
data = np.random.randn(1000, 100).astype(np.float32)

# Process
results = process_in_batches(data, batch_size=64, model_fn=simple_model)
print(f"Processed {results.shape[0]} samples, output shape: {results.shape}")
```

---

## Attention Mechanism

### Scaled Dot-Product Attention

```python
import pyflame as pf
import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    """Compute scaled dot-product attention.

    Args:
        query: [batch, seq_len, d_k]
        key: [batch, seq_len, d_k]
        value: [batch, seq_len, d_v]
        mask: Optional attention mask

    Returns:
        Attention output [batch, seq_len, d_v]
    """
    d_k = query.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    # Compute attention scores: [batch, seq_len, seq_len]
    scores = (query @ key.transpose(1, 2)) * scale

    # Apply mask if provided (set masked positions to large negative value)
    # Note: mask support would be added here

    # Softmax over last dimension
    attention_weights = pf.softmax(scores, dim=-1)

    # Apply attention to values
    output = attention_weights @ value

    return output, attention_weights


# Example usage
batch_size = 8
seq_len = 32
d_model = 64

# Create Q, K, V
Q = pf.randn([batch_size, seq_len, d_model])
K = pf.randn([batch_size, seq_len, d_model])
V = pf.randn([batch_size, seq_len, d_model])

# Compute attention
output, weights = scaled_dot_product_attention(Q, K, V)
pf.eval(output, weights)

print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Weights sum per query (should be 1): {weights.numpy()[0, 0].sum():.6f}")
```

### Multi-Head Attention

```python
import pyflame as pf
import numpy as np

class MultiHeadAttention:
    """Multi-head attention mechanism."""

    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / d_model)

        # Projection weights
        self.W_q = pf.randn([d_model, d_model]) * scale
        self.W_k = pf.randn([d_model, d_model]) * scale
        self.W_v = pf.randn([d_model, d_model]) * scale
        self.W_o = pf.randn([d_model, d_model]) * scale

    def forward(self, x):
        """Apply multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Project to Q, K, V
        Q = x @ self.W_q  # [batch, seq, d_model]
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head: [batch, seq, n_heads, d_k]
        # Then transpose to [batch, n_heads, seq, d_k]
        Q = Q.reshape([batch_size, seq_len, self.n_heads, self.d_k]).transpose(1, 2)
        K = K.reshape([batch_size, seq_len, self.n_heads, self.d_k]).transpose(1, 2)
        V = V.reshape([batch_size, seq_len, self.n_heads, self.d_k]).transpose(1, 2)

        # Scaled dot-product attention per head
        scale = 1.0 / np.sqrt(self.d_k)
        scores = (Q @ K.transpose(2, 3)) * scale  # [batch, n_heads, seq, seq]
        attn = pf.softmax(scores, dim=-1)
        context = attn @ V  # [batch, n_heads, seq, d_k]

        # Concatenate heads: [batch, seq, d_model]
        context = context.transpose(1, 2).reshape([batch_size, seq_len, self.d_model])

        # Final projection
        output = context @ self.W_o

        return output


# Example
mha = MultiHeadAttention(d_model=256, n_heads=8)
x = pf.randn([4, 32, 256])  # [batch, seq, d_model]
output = mha.forward(x)
pf.eval(output)

print(f"Multi-head attention output shape: {output.shape}")
```

---

## Distributed Computation with Layouts

### Row-Partitioned Matrix Multiply

```python
import pyflame as pf

# Large matrices distributed across PEs
rows = 4096
cols = 4096
n_pes = 16

# Distribute rows across PEs
row_layout = pf.MeshLayout.row_partition(n_pes)

A = pf.randn([rows, cols], layout=row_layout)
B = pf.randn([cols, rows], layout=pf.MeshLayout.single_pe())

# Matrix multiply - each PE computes its portion
C = A @ B

# Evaluate
pf.eval(C)
print(f"Distributed result shape: {C.shape}")
```

### 2D Grid Distribution

```python
import pyflame as pf

# Very large matrix on 2D grid of PEs
size = 8192
grid_rows = 8
grid_cols = 8  # 64 PEs total

# Create 2D distributed tensor
grid_layout = pf.MeshLayout.grid(grid_rows, grid_cols)
X = pf.randn([size, size], layout=grid_layout)
Y = pf.randn([size, size], layout=grid_layout)

# Operations on distributed tensors
Z = X + Y        # Element-wise (local to each PE)
W = pf.relu(Z)   # Activation (local to each PE)

# Reduction across all PEs
total = W.sum()

pf.eval(total)
print(f"Sum across {grid_rows * grid_cols} PEs: {total.numpy()}")
```

---

## CSL Code Generation

### Generating CSL for a Simple Model

```python
import pyflame as pf

# Define computation
input_size = 1024
hidden_size = 512
output_size = 256

x = pf.randn([32, input_size])
w1 = pf.randn([input_size, hidden_size]) * 0.01
b1 = pf.zeros([hidden_size])
w2 = pf.randn([hidden_size, output_size]) * 0.01
b2 = pf.zeros([output_size])

# Build graph
h = pf.relu(x @ w1 + b1)
y = h @ w2 + b2
output = pf.softmax(y, dim=1)

# Generate CSL code
options = pf.CodeGenOptions()
options.target = "wse2"
options.output_dir = "./generated_csl"
options.emit_comments = True

result = pf.compile_to_csl(output, options)

if result.success:
    print(f"Generated {len(result.generated_files)} files:")
    for f in result.generated_files:
        print(f"  - {f}")
else:
    print(f"Error: {result.error_message}")
```

### Inspecting Generated Code

```python
import pyflame as pf

# Simple computation
a = pf.randn([100, 100])
b = pf.randn([100, 100])
c = pf.relu(a @ b)
d = c.sum()

# Generate and inspect
generator = pf.CSLCodeGenerator()
graph = pf.get_graph(d)

options = pf.CodeGenOptions()
options.emit_comments = True

# Get sources without writing files
sources = generator.generate_source(graph, options)

for filename, content in sources.items():
    print(f"\n=== {filename} ===")
    print(content[:500])  # Print first 500 chars
    print("...")
```

---

## Working with NumPy

### Data Pipeline Integration

```python
import numpy as np
import pyflame as pf

def preprocess_numpy(data):
    """Preprocess data with NumPy."""
    # Normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    normalized = (data - mean) / std

    # Ensure float32
    return normalized.astype(np.float32)

def pyflame_forward(x_np, weights, bias):
    """Run forward pass with PyFlame."""
    x = pf.from_numpy(x_np)
    w = pf.from_numpy(weights)
    b = pf.from_numpy(bias)

    output = pf.relu(x @ w + b)
    pf.eval(output)

    return output.numpy()

def postprocess_numpy(output):
    """Post-process with NumPy."""
    # Apply custom numpy operations
    return np.clip(output, 0, 10)


# Full pipeline
raw_data = np.random.randn(100, 50)
weights = np.random.randn(50, 25).astype(np.float32) * 0.1
bias = np.zeros(25, dtype=np.float32)

# Pipeline
normalized = preprocess_numpy(raw_data)
output = pyflame_forward(normalized, weights, bias)
final = postprocess_numpy(output)

print(f"Input shape: {raw_data.shape}")
print(f"Output shape: {final.shape}")
```

### Comparison with NumPy Results

```python
import numpy as np
import pyflame as pf

# Create matching data
np.random.seed(42)
a_np = np.random.randn(100, 50).astype(np.float32)
b_np = np.random.randn(50, 30).astype(np.float32)

# NumPy computation
c_np = np.maximum(0, a_np @ b_np)  # ReLU(matmul)

# PyFlame computation
a_pf = pf.from_numpy(a_np)
b_pf = pf.from_numpy(b_np)
c_pf = pf.relu(a_pf @ b_pf)
pf.eval(c_pf)

# Compare
c_result = c_pf.numpy()
max_diff = np.abs(c_np - c_result).max()
print(f"Maximum difference: {max_diff:.2e}")
np.testing.assert_allclose(c_np, c_result, rtol=1e-5)
print("Results match!")
```

---

## Complete Example: Image Classification Pipeline

```python
import numpy as np
import pyflame as pf

class ImageClassifier:
    """Simple image classification model."""

    def __init__(self, input_dim=784, hidden_dims=[256, 128], num_classes=10):
        self.layers = []
        dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            w = pf.randn([dims[i], dims[i+1]]) * scale
            b = pf.zeros([dims[i+1]])
            self.layers.append((w, b))

    def forward(self, x):
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = pf.relu(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return pf.softmax(logits, dim=1)


def load_batch(batch_idx, batch_size=64):
    """Simulate loading a batch of images."""
    # In practice, load from file/database
    images = np.random.randn(batch_size, 784).astype(np.float32)
    labels = np.random.randint(0, 10, batch_size)
    return images, labels


def compute_accuracy(probs, labels):
    """Compute classification accuracy."""
    predictions = probs.argmax(axis=1)
    return (predictions == labels).mean()


# Create model
model = ImageClassifier()

# Process multiple batches
n_batches = 10
accuracies = []

for batch_idx in range(n_batches):
    # Load data
    images, labels = load_batch(batch_idx)

    # Convert to PyFlame
    x = pf.from_numpy(images)

    # Forward pass
    probs = model.predict(x)
    pf.eval(probs)

    # Compute accuracy
    acc = compute_accuracy(probs.numpy(), labels)
    accuracies.append(acc)

    if batch_idx % 5 == 0:
        print(f"Batch {batch_idx}: accuracy = {acc:.2%}")

print(f"\nAverage accuracy: {np.mean(accuracies):.2%}")
```

---

## Next Steps

- Review the [API Reference](api_reference.md) for complete function documentation
- Read [Best Practices](best_practices.md) for optimization tips
- See [Integration Guide](integration_guide.md) for adding PyFlame to your project
