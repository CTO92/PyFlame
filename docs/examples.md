# PyFlame Examples

**PyFlame Version:** Pre-Release Alpha 1.0

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs described here are subject to change.

---

This document provides practical examples of using PyFlame for common deep learning tasks.

---

## Table of Contents

### Core Operations
1. [Basic Tensor Operations](#basic-tensor-operations)
2. [Batch Processing](#batch-processing)
3. [Distributed Computation with Layouts](#distributed-computation-with-layouts)
4. [CSL Code Generation](#csl-code-generation)
5. [Working with NumPy](#working-with-numpy)

### Neural Networks (Phase 2)
6. [Building Models with nn.Module](#building-models-with-nnmodule)
7. [Convolutional Neural Networks](#convolutional-neural-networks)
8. [Training with Optimizers](#training-with-optimizers)
9. [Learning Rate Scheduling](#learning-rate-scheduling)
10. [Transformer Attention](#transformer-attention)
11. [Complete Training Example](#complete-training-example)

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

## Building Models with nn.Module

### Simple Linear Model

```python
import pyflame as pf
from pyflame import nn

# Create a simple linear classifier
model = nn.Linear(784, 10)

# Forward pass
x = pf.randn([32, 784])  # Batch of 32 images
logits = model(x)
probs = pf.softmax(logits, dim=1)

pf.eval(probs)
print(f"Output shape: {probs.shape}")  # [32, 10]
```

### Custom Model with Sequential

```python
import pyflame as pf
from pyflame import nn

# Build MLP with Sequential
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Linear(256, 128),
    nn.Linear(128, 10)
])

# Use the model
x = pf.randn([64, 784])
output = model(x)
pf.eval(output)

print(f"Model has {len(model.parameters())} parameter tensors")
```

### Training and Evaluation Modes

```python
import pyflame as pf
from pyflame import nn

# Model with dropout
class DropoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = self.register_module("fc1", nn.Linear(100, 50))
        self.dropout = self.register_module("dropout", nn.Dropout(0.5))
        self.fc2 = self.register_module("fc2", nn.Linear(50, 10))

    def forward(self, x):
        x = pf.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DropoutMLP()

# Training mode (dropout active)
model.train()
train_output = model(pf.randn([32, 100]))

# Evaluation mode (dropout disabled)
model.eval()
eval_output = model(pf.randn([32, 100]))
```

---

## Convolutional Neural Networks

### Simple CNN

```python
import pyflame as pf
from pyflame import nn

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.register_module("conv1",
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)))
        self.pool1 = self.register_module("pool1",
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.conv2 = self.register_module("conv2",
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)))
        self.pool2 = self.register_module("pool2",
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.gap = self.register_module("gap", nn.GlobalAvgPool2d())
        self.fc = self.register_module("fc", nn.Linear(64, num_classes))

    def forward(self, x):
        # x: [N, 1, 28, 28]
        x = pf.relu(self.conv1(x))  # [N, 32, 28, 28]
        x = self.pool1(x)           # [N, 32, 14, 14]
        x = pf.relu(self.conv2(x))  # [N, 64, 14, 14]
        x = self.pool2(x)           # [N, 64, 7, 7]
        x = self.gap(x)             # [N, 64, 1, 1]
        x = x.reshape([x.shape[0], -1])  # [N, 64]
        x = self.fc(x)              # [N, 10]
        return x

model = SimpleCNN()
x = pf.randn([16, 1, 28, 28])  # Batch of MNIST-like images
output = model(x)
pf.eval(output)
print(f"CNN output shape: {output.shape}")  # [16, 10]
```

### CNN with BatchNorm

```python
import pyflame as pf
from pyflame import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = self.register_module("conv",
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=(1, 1)))
        self.bn = self.register_module("bn", nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return pf.relu(self.bn(self.conv(x)))

# Stack conv blocks
model = nn.Sequential([
    ConvBlock(3, 64),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    ConvBlock(64, 128),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    ConvBlock(128, 256),
    nn.AdaptiveAvgPool2d((1, 1))
])

x = pf.randn([8, 3, 32, 32])
features = model(x)
pf.eval(features)
print(f"Feature shape: {features.shape}")  # [8, 256, 1, 1]
```

---

## Training with Optimizers

### Basic Training Loop

```python
import pyflame as pf
from pyflame import nn
from pyflame import optim
import numpy as np

# Create model and optimizer
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Linear(256, 10)
])

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    # Simulated batch
    x = pf.randn([64, 784])
    targets = pf.tensor(np.random.randint(0, 10, 64))

    # Forward pass
    model.train()
    logits = model(x)
    loss = loss_fn(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    pf.autograd.backward(loss)

    # Update weights
    optimizer.step()

    pf.eval(loss)
    print(f"Epoch {epoch}: loss = {loss.numpy():.4f}")
```

### SGD with Momentum

```python
import pyflame as pf
from pyflame import nn, optim

model = nn.Linear(100, 10)

# SGD with momentum and weight decay
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# Training step
x = pf.randn([32, 100])
y = pf.randn([32, 10])

output = model(x)
loss = nn.functional.mse_loss(output, y)

optimizer.zero_grad()
pf.autograd.backward(loss)
optimizer.step()
```

### AdamW for Transformers

```python
import pyflame as pf
from pyflame import nn, optim

# Transformer-style model
model = nn.Sequential([
    nn.Linear(512, 2048),
    nn.Linear(2048, 512)
])

# AdamW with cosine learning rate schedule
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.98
)
```

---

## Learning Rate Scheduling

### Step Decay

```python
import pyflame as pf
from pyflame import nn, optim

model = nn.Linear(100, 10)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Decay LR by 0.1 every 30 epochs
scheduler = optim.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    # Train...
    scheduler.step()
    print(f"Epoch {epoch}: LR = {scheduler.get_lr():.6f}")
```

### Cosine Annealing

```python
import pyflame as pf
from pyflame import nn, optim

model = nn.Linear(100, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Cosine annealing over 100 epochs
scheduler = optim.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

for epoch in range(100):
    # Train...
    scheduler.step()
```

### One-Cycle Learning Rate

```python
import pyflame as pf
from pyflame import nn, optim

model = nn.Linear(100, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_steps = 1000 * 10  # 10 epochs, 1000 batches each
scheduler = optim.OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=total_steps,
    pct_start=0.3  # 30% warmup
)

for step in range(total_steps):
    # Train step...
    scheduler.step()
```

### Reduce on Plateau

```python
import pyflame as pf
from pyflame import nn, optim

model = nn.Linear(100, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Reduce when metric stops decreasing
    factor=0.5,      # Multiply LR by 0.5
    patience=5,      # Wait 5 epochs before reducing
    min_lr=1e-7
)

for epoch in range(100):
    train_loss = train(model)
    val_loss = validate(model)

    # Pass validation metric to scheduler
    scheduler.step(val_loss)
```

---

## Transformer Attention

### Using MultiheadAttention

```python
import pyflame as pf
from pyflame import nn

# Create attention layer
embed_dim = 512
num_heads = 8
mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

# Self-attention
seq_len, batch_size = 32, 16
x = pf.randn([seq_len, batch_size, embed_dim])  # [seq, batch, embed]

mha.train()
output, attn_weights = mha.forward(x, x, x, need_weights=True)
pf.eval(output, attn_weights)

print(f"Output shape: {output.shape}")  # [32, 16, 512]
print(f"Attention weights: {attn_weights.shape}")  # [16, 32, 32]
```

### Transformer Encoder Block

```python
import pyflame as pf
from pyflame import nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = self.register_module("attn",
            nn.MultiheadAttention(d_model, n_heads, dropout))
        self.norm1 = self.register_module("norm1", nn.LayerNorm([d_model]))
        self.norm2 = self.register_module("norm2", nn.LayerNorm([d_model]))
        self.ff = self.register_module("ff", nn.Sequential([
            nn.Linear(d_model, d_ff),
            nn.Linear(d_ff, d_model)
        ]))
        self.dropout = self.register_module("dropout", nn.Dropout(dropout))

    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ff_out = self.ff(pf.gelu(x))
        x = self.norm2(x + self.dropout(ff_out))

        return x

# Stack transformer blocks
d_model, n_heads, d_ff = 256, 8, 1024
encoder = nn.Sequential([
    TransformerBlock(d_model, n_heads, d_ff),
    TransformerBlock(d_model, n_heads, d_ff),
    TransformerBlock(d_model, n_heads, d_ff)
])

x = pf.randn([50, 8, d_model])  # [seq, batch, d_model]
output = encoder(x)
pf.eval(output)
print(f"Encoder output: {output.shape}")
```

---

## Complete Training Example

### Training a Classifier

```python
import pyflame as pf
from pyflame import nn, optim
import numpy as np

# ============================================
# Model Definition
# ============================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = self.register_module("fc1", nn.Linear(input_dim, hidden_dim))
        self.bn1 = self.register_module("bn1", nn.BatchNorm1d(hidden_dim))
        self.dropout = self.register_module("dropout", nn.Dropout(0.3))
        self.fc2 = self.register_module("fc2", nn.Linear(hidden_dim, hidden_dim))
        self.bn2 = self.register_module("bn2", nn.BatchNorm1d(hidden_dim))
        self.fc3 = self.register_module("fc3", nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = pf.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = pf.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# ============================================
# Setup
# ============================================
model = MLP(input_dim=784, hidden_dim=256, num_classes=10)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=100)
loss_fn = nn.CrossEntropyLoss()

# ============================================
# Training Loop
# ============================================
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    model.train()

    # Simulated training data
    x_train = pf.randn([batch_size, 784])
    y_train = pf.tensor(np.random.randint(0, 10, batch_size))

    # Forward
    logits = model(x_train)
    loss = loss_fn(logits, y_train)

    # Backward
    optimizer.zero_grad()
    pf.autograd.backward(loss)

    # Update
    optimizer.step()
    scheduler.step()

    # Validation
    if epoch % 10 == 0:
        model.eval()
        with pf.autograd.no_grad():
            x_val = pf.randn([batch_size, 784])
            y_val = np.random.randint(0, 10, batch_size)

            val_logits = model(x_val)
            pf.eval(val_logits, loss)

            predictions = val_logits.numpy().argmax(axis=1)
            accuracy = (predictions == y_val).mean()

            print(f"Epoch {epoch}: loss={loss.numpy():.4f}, "
                  f"acc={accuracy:.2%}, lr={scheduler.get_lr():.6f}")

# ============================================
# Save and Load Model
# ============================================
# Save state
state = model.state_dict()
optim_state = optimizer.state_dict()

# Load state
new_model = MLP(784, 256, 10)
new_model.load_state_dict(state)

print("Training complete!")
```

---

## Developer Tools Examples

### Profiling a Model

```python
import pyflame as pf
from pyflame import nn
from pyflame.tools import Profiler

# Create a model
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Linear(256, 128),
    nn.Linear(128, 10)
])

# Profile with memory tracking
profiler = Profiler(track_memory=True)

with profiler:
    for _ in range(10):
        x = pf.randn([64, 784])
        output = model(x)
        pf.eval(output)

# Get results
result = profiler.get_result()
print(result.summary())

# Export to Chrome trace format (open with chrome://tracing)
profiler.export_chrome_trace("model_profile.json")
```

### Graph Visualization

```python
import pyflame as pf
from pyflame import nn
from pyflame.tools import visualize_model, GraphVisualizer

# Create model
model = nn.Sequential([
    nn.Linear(100, 50),
    nn.Linear(50, 10)
])

# Quick visualization
example_input = pf.randn([1, 100])
visualize_model(model, example_input, "model_graph.svg")

# Custom visualization options
viz = GraphVisualizer(
    graph=None,
    show_shapes=True,
    show_dtypes=True,
    rankdir="LR"  # Left-to-right layout
)
```

### Debugging with Breakpoints

```python
from pyflame.tools import PyFlameDebugger

# Create debugger
debugger = PyFlameDebugger()

# Set breakpoints on specific operations
debugger.set_breakpoint("matmul")
debugger.set_breakpoint("relu", condition="output.max() > 10")

# Watch tensors during execution
debugger.watch("activations", lambda t: f"mean={t.mean():.4f}")

with debugger:
    output = model(input_data)
```

---

## Model Serving Examples

### InferenceEngine with Caching

```python
import pyflame as pf
from pyflame import nn
from pyflame.serving import InferenceEngine, InferenceConfig

# Create model
model = nn.Sequential([
    nn.Linear(100, 50),
    nn.Linear(50, 10)
])
model.eval()

# Configure engine with caching
config = InferenceConfig(
    batch_size=32,
    enable_caching=True,
    cache_size=1000,
    warmup_iterations=5
)

# Create engine
engine = InferenceEngine(model, config)

# Warmup
example = pf.randn([1, 100])
engine.warmup(example)

# Run inference
for i in range(100):
    input_data = pf.randn([32, 100])
    output = engine.infer(input_data)

# Check statistics
stats = engine.get_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Average latency: {stats.average_time_ms:.2f}ms")
print(f"Throughput: {stats.throughput:.0f} req/s")
print(f"Cache hit rate: {stats.cache_hit_rate:.1%}")
```

### REST API Model Server

```python
from pyflame import nn
from pyflame.serving import ModelServer, ServerConfig

# Create and prepare model
model = nn.Sequential([
    nn.Linear(100, 50),
    nn.Linear(50, 10)
])
model.eval()

# Configure server
config = ServerConfig(
    host="0.0.0.0",
    port=8000,
    workers=4,
    max_batch_size=64,
    enable_cors=True
)

# Start server
server = ModelServer(model, config)
print("Starting server at http://localhost:8000")
server.start()  # Blocking

# Endpoints available:
# POST /v1/predict - Run inference
# GET  /v1/health  - Health check
# GET  /v1/info    - Model info
# GET  /v1/stats   - Server statistics
```

### Client for Remote Inference

```python
from pyflame.serving import ModelClient, ClientConfig
import numpy as np

# Configure client
config = ClientConfig(
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0
)

# Create client
client = ModelClient("http://localhost:8000", config)

# Check server health
if client.health_check():
    print("Server is healthy")

# Single prediction
input_data = np.random.randn(1, 100).tolist()
output = client.predict(input_data)
print(f"Prediction: {output}")

# Batch prediction
inputs = [np.random.randn(100).tolist() for _ in range(10)]
outputs = client.predict_batch(inputs)
print(f"Batch outputs: {len(outputs)} predictions")

# Get server info
info = client.get_info()
print(f"Model info: {info}")
```

---

## Benchmarking Examples

### Benchmark Model Performance

```python
import pyflame as pf
from pyflame import nn
from pyflame.benchmarks import BenchmarkRunner, BenchmarkConfig

# Create model
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Linear(256, 10)
])

# Configure benchmark
config = BenchmarkConfig(
    warmup_iterations=10,
    benchmark_iterations=100,
    batch_sizes=[1, 8, 32, 64, 128],
    enable_memory_tracking=True
)

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run_model_benchmark(
    name="mlp_classifier",
    model=model,
    input_shape=[784]
)

# Print results
print("\nBenchmark Results:")
print("-" * 60)
for r in results:
    print(f"Batch {r.batch_size:4d}: "
          f"{r.latency_ms:8.2f}ms, "
          f"{r.throughput:8.0f} samples/sec, "
          f"{r.memory_mb:6.1f} MB")

# Export results
runner.export_json("benchmark_results.json")
runner.export_csv("benchmark_results.csv")
```

### Quick Benchmark

```python
from pyflame import nn
from pyflame.benchmarks import benchmark

model = nn.Linear(1000, 100)

# One-liner benchmark
results = benchmark(
    model,
    input_shape=[1000],
    batch_sizes=[1, 16, 64],
    iterations=50,
    print_results=True
)
```

### Compare Benchmark Results

```python
from pyflame.benchmarks import BenchmarkReport, compare_results

# Load previous benchmark
baseline = BenchmarkReport.load_json("baseline_results.json")

# Run new benchmark
new_results = runner.run_model_benchmark("model_v2", new_model, [784])
new_report = BenchmarkReport(name="v2", results=new_results)

# Compare
comparison = compare_results(baseline.results, new_results)

print("\nPerformance Comparison:")
for name, diff in comparison.items():
    if diff['speedup'] > 1:
        print(f"  {name}: {diff['speedup']:.2f}x faster")
    else:
        print(f"  {name}: {1/diff['speedup']:.2f}x slower")
```

### Benchmark Operations

```python
import pyflame as pf
from pyflame.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()

# Benchmark matrix multiplication
result = runner.run_operation_benchmark(
    name="matmul_1024x1024",
    operation=lambda x, y: x @ y,
    input_shapes=[[1024, 1024], [1024, 1024]]
)

print(f"MatMul 1024x1024: {result.latency_ms:.2f}ms")

# Benchmark with timing context
from pyflame.benchmarks import timed

with timed() as t:
    for _ in range(100):
        x = pf.randn([256, 256])
        y = pf.relu(x @ x)
        pf.eval(y)

print(f"100 iterations: {t.elapsed_ms:.2f}ms")
print(f"Average: {t.elapsed_ms / 100:.2f}ms per iteration")
```

---

## Integration Examples

### Weights & Biases Logging

```python
import pyflame as pf
from pyflame import nn, optim
from pyflame.training import Trainer, TrainerConfig
from pyflame.integrations import WandbCallback

# Create model
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Linear(256, 10)
])

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Setup W&B callback
wandb_callback = WandbCallback(
    project="pyflame-experiments",
    name="mlp-classifier",
    config={
        "architecture": "mlp",
        "learning_rate": 0.001,
        "batch_size": 64
    },
    log_model=True,
    log_gradients=True
)

# Setup trainer with callback
config = TrainerConfig(max_epochs=50, log_interval=10)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    config=config,
    callbacks=[wandb_callback]
)

# Train
trainer.fit(train_loader, val_loader)

# W&B run is automatically finished
```

### MLflow Experiment Tracking

```python
import pyflame as pf
from pyflame import nn, optim
from pyflame.training import Trainer, TrainerConfig
from pyflame.integrations import MLflowCallback

# Create model
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Linear(256, 10)
])

# Setup MLflow callback
mlflow_callback = MLflowCallback(
    experiment_name="pyflame-experiments",
    run_name="mlp-v1",
    tracking_uri="http://localhost:5000",
    log_models=True
)

# Log hyperparameters
mlflow_callback.log_params({
    "model_type": "mlp",
    "hidden_size": 256,
    "learning_rate": 0.001
})

# Train with callback
trainer = Trainer(
    model=model,
    optimizer=optim.Adam(model.parameters()),
    loss_fn=nn.CrossEntropyLoss(),
    callbacks=[mlflow_callback]
)

trainer.fit(train_loader)
```

### ONNX Export

```python
import pyflame as pf
from pyflame import nn
from pyflame.integrations import ONNXExporter, export_onnx

# Create model
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Linear(256, 10)
])
model.eval()

# Quick export
example_input = pf.randn([1, 784])
export_onnx(model, example_input, "model.onnx")

# Advanced export with options
exporter = ONNXExporter(opset_version=17)
exporter.export(
    model,
    example_input,
    "model_dynamic.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)

# Verify exported model
exporter.verify(model, example_input, "model_dynamic.onnx")
print("ONNX export verified successfully!")
```

---

## Custom Extensions Examples

### Custom Operator

```python
import pyflame as pf
from pyflame.extend import custom_op, register_custom_op

# Using decorator
@custom_op(name="swish")
def swish(x):
    """Swish activation: x * sigmoid(x)"""
    return x * pf.sigmoid(x)

# Use the custom op
x = pf.randn([32, 100])
y = swish(x)
pf.eval(y)

# With backward pass
def mish_forward(x):
    return x * pf.tanh(pf.softplus(x))

def mish_backward(grad_output, x):
    # Approximate gradient
    return grad_output * pf.tanh(pf.softplus(x))

register_custom_op(
    name="mish",
    forward_fn=mish_forward,
    backward_fn=mish_backward
)
```

### Custom Autograd Function

```python
from pyflame.extend import AutogradFunction, FunctionContext

class Clamp(AutogradFunction):
    """Custom clamp function with gradient."""

    @staticmethod
    def forward(ctx: FunctionContext, x, min_val, max_val):
        ctx.save_for_backward(x)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return x.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx: FunctionContext, grad_output):
        x, = ctx.saved_tensors
        # Gradient is 1 where x is within bounds, 0 otherwise
        mask = (x >= ctx.min_val) & (x <= ctx.max_val)
        return grad_output * mask, None, None

# Use custom function
x = pf.randn([100])
y = Clamp.apply(x, -1.0, 1.0)
```

### Creating a Plugin

```python
from pyflame.extend import Plugin, PluginInfo, register_plugin

class MyActivationsPlugin(Plugin):
    """Plugin providing custom activation functions."""

    @classmethod
    def get_info(cls) -> PluginInfo:
        return PluginInfo(
            name="my-activations",
            version="1.0.0",
            author="Your Name",
            description="Custom activation functions for PyFlame"
        )

    def on_load(self):
        print("Loading custom activations...")
        # Register custom ops when plugin loads
        for name, fn in self.get_custom_ops().items():
            register_custom_op(name, fn)

    def on_unload(self):
        print("Unloading custom activations...")

    def get_custom_ops(self):
        return {
            "hard_swish": lambda x: x * pf.relu(x + 3) / 6,
            "quick_gelu": lambda x: x * pf.sigmoid(1.702 * x)
        }

# Register and use
register_plugin(MyActivationsPlugin)

from pyflame.extend import load_plugin
load_plugin("my-activations")
```

### Plugin Manager

```python
from pyflame.extend import PluginManager

# Get plugin manager (singleton)
manager = PluginManager()

# Discover plugins in a directory
manager.discover_plugins("./plugins")

# List available plugins
print("Available plugins:")
for name in manager.list_plugins():
    plugin = manager.get_plugin(name)
    info = plugin.get_info()
    print(f"  - {info.name} v{info.version}: {info.description}")

# Load a specific plugin
manager.load_plugin("my-activations")

# Unload when done
manager.unload_plugin("my-activations")
```

---

## Next Steps

- Review the [API Reference](api_reference.md) for complete function documentation
- Read [Best Practices](best_practices.md) for optimization tips
- See [Integration Guide](integration_guide.md) for adding PyFlame to your project
