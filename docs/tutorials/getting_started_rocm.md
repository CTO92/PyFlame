# Getting Started with PyFlame on AMD GPUs

This tutorial will guide you through setting up PyFlame with the ROCm backend to run deep learning workloads on AMD GPUs.

## Prerequisites

Before starting, ensure you have:

- An AMD GPU with ROCm support (MI100, MI200, MI300, or RX 6000/7000 series)
- Linux operating system (Ubuntu 20.04/22.04 recommended)
- ROCm 5.4 or later installed
- Python 3.8 or later

## Step 1: Verify ROCm Installation

First, verify that ROCm is properly installed and can detect your GPU:

```bash
# Check ROCm installation
rocm-smi

# Expected output:
# ======================= ROCm System Management Interface =======================
# GPU  Temp   AvgPwr  SCLK    MCLK    VRAM%  GPU%  ...
# 0    45c    35W     800Mhz  1200Mhz  0%     0%   ...

# Get detailed GPU info
/opt/rocm/bin/rocminfo | grep -A 5 "Name:"
```

If you see your GPU listed, ROCm is working correctly.

## Step 2: Build PyFlame with ROCm Support

```bash
# Clone PyFlame
git clone https://github.com/your-org/pyflame.git
cd pyflame

# Create build directory
mkdir build && cd build

# Configure with ROCm
cmake -DPYFLAME_USE_ROCM=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Build (use all available cores)
make -j$(nproc)

# Install Python package
cd ..
pip install -e python/
```

## Step 3: Verify PyFlame ROCm Support

```python
import pyflame as pf

# Check ROCm availability
print(f"ROCm available: {pf.rocm_is_available()}")
print(f"Number of GPUs: {pf.rocm_device_count()}")

# Get GPU info
if pf.rocm_is_available():
    info = pf.rocm_get_device_info(0)
    print(f"\nGPU 0:")
    print(f"  Name: {info['name']}")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Memory: {info['total_memory'] / 1e9:.1f} GB")
    print(f"  Compute Units: {info['compute_units']}")
```

Expected output:
```
ROCm available: True
Number of GPUs: 1

GPU 0:
  Name: AMD Instinct MI100
  Architecture: gfx908
  Memory: 32.0 GB
  Compute Units: 120
```

## Step 4: Your First GPU Computation

Let's run a simple matrix multiplication on the GPU:

```python
import pyflame as pf
import time

# Select ROCm backend
pf.set_device('rocm')
print(f"Using device: {pf.get_device()}")

# Create random matrices
size = 4096
a = pf.randn(size, size)
b = pf.randn(size, size)

# Warm up (first run compiles kernels)
c = pf.matmul(a, b)
pf.synchronize()

# Benchmark
iterations = 100
start = time.time()
for _ in range(iterations):
    c = pf.matmul(a, b)
pf.synchronize()
elapsed = time.time() - start

# Calculate TFLOPS
flops = 2 * size * size * size * iterations
tflops = flops / elapsed / 1e12
print(f"\nMatrix multiply {size}x{size}:")
print(f"  Time: {elapsed:.3f}s for {iterations} iterations")
print(f"  Performance: {tflops:.2f} TFLOPS")
```

## Step 5: Running a Neural Network

Here's how to train a simple neural network on an AMD GPU:

```python
import pyflame as pf
import pyflame.nn as nn
import pyflame.optim as optim
import numpy as np

# Use AMD GPU
pf.set_device('rocm')

# Define a simple MLP for MNIST
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = pf.relu(self.fc1(x))
        x = pf.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model and optimizer
model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate dummy data
batch_size = 64
num_batches = 100

print("Training on AMD GPU...")
for batch in range(num_batches):
    # Random input and targets
    x = pf.randn(batch_size, 784)
    targets = pf.randint(0, 10, (batch_size,))

    # Forward pass
    logits = model(x)
    loss = pf.cross_entropy(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 20 == 0:
        print(f"  Batch {batch}: loss = {loss.item():.4f}")

print("Training complete!")
```

## Step 6: Convolution Operations

For image-based models, you'll use convolution layers:

```python
import pyflame as pf

pf.set_device('rocm')

# Create input: batch of 32 RGB images, 224x224
input = pf.randn(32, 3, 224, 224)

# Convolution weights: 64 output channels, 3 input channels, 7x7 kernel
weight = pf.randn(64, 3, 7, 7)

# Apply convolution with stride=2, padding=3 (like ResNet first layer)
output = pf.conv2d(input, weight, stride=2, padding=3)

print(f"Input shape: {input.shape}")
print(f"Output shape: {output.shape}")  # [32, 64, 112, 112]

# Apply batch normalization
gamma = pf.ones(64)
beta = pf.zeros(64)
mean = pf.zeros(64)
var = pf.ones(64)
output = pf.batch_norm(output, mean, var, gamma, beta, training=False)

# Apply ReLU activation
output = pf.relu(output)
```

## Step 7: Performance Tips

### Enable Auto-tuning

For best convolution performance, enable MIOpen auto-tuning:

```python
from pyflame.backend.rocm import get_tuning_cache

# Enable exhaustive algorithm search
cache = get_tuning_cache()
cache.enable_exhaustive_search(True)

# Optionally save tuning results to disk
cache.set_cache_file('/path/to/miopen_cache.bin')
```

### Warm Up Memory Pools

Reduce first-run latency:

```python
from pyflame.backend.rocm import warmup_memory_pool

# Warmup for batch size 32, medium-sized model
warmup_memory_pool(batch_size=32, model_size='medium')
```

### Use Appropriate Batch Sizes

Larger batches better utilize GPU parallelism:

```python
# Prefer larger batches when possible
# Good: batch_size=64 or 128
# May not fully utilize GPU: batch_size=8
```

## Step 8: Running Benchmarks

PyFlame includes a benchmark suite:

```bash
# Run all benchmarks
python benchmarks/rocm_benchmark.py --all

# Run specific benchmarks
python benchmarks/rocm_benchmark.py --matmul --conv

# Typical output:
# ==============================================================
# ROCm Performance Benchmark Suite
# ==============================================================
# Device: AMD Instinct MI100
# Memory: 32.0 GB total, 31.2 GB free
#
# === Matrix Multiplication ===
#     512x512x512:    0.052 ms |  5.04 TFLOPS
#    1024x1024x1024:  0.198 ms | 10.84 TFLOPS
#    2048x2048x2048:  1.123 ms | 15.32 TFLOPS
```

## Troubleshooting

### "ROCm not available"

```python
# Check if HIP can see your GPU
import subprocess
result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
print(result.stdout)
```

If no GPU is shown:
1. Check that ROCm is installed: `apt list --installed | grep rocm`
2. Check permissions: Add your user to the `video` and `render` groups
3. Reboot after ROCm installation

### Slow First Run

The first run is slower because MIOpen needs to find optimal algorithms:

```python
# Expected: First forward pass takes longer
import time

pf.set_device('rocm')
x = pf.randn(32, 64, 56, 56)
w = pf.randn(128, 64, 3, 3)

# First run - algorithm search
start = time.time()
y = pf.conv2d(x, w, padding=1)
pf.synchronize()
print(f"First run: {time.time() - start:.3f}s")

# Subsequent runs - cached algorithm
start = time.time()
y = pf.conv2d(x, w, padding=1)
pf.synchronize()
print(f"Second run: {time.time() - start:.3f}s")
```

### Out of Memory

```python
# Check GPU memory usage
info = pf.device_info()
print(f"Free: {info['free_memory'] / 1e9:.1f} GB")
print(f"Total: {info['total_memory'] / 1e9:.1f} GB")

# Clear memory pool if needed
from pyflame.backend.rocm import get_memory_manager
get_memory_manager().clear_pool()
```

## Next Steps

- Read the [ROCm Backend Guide](../backends/rocm.md) for detailed configuration options
- Check the [API Reference](../api/rocm_api.md) for complete API documentation
- Explore the `examples/` directory for more complex models
- Join our community forum for support

## Quick Reference

```python
import pyflame as pf

# Device management
pf.set_device('rocm')           # Use AMD GPU
pf.set_device('rocm:1')         # Use specific GPU
pf.get_device()                 # Get current device
pf.device_info()                # Get device info
pf.synchronize()                # Wait for GPU operations

# ROCm-specific
pf.rocm_is_available()          # Check availability
pf.rocm_device_count()          # Number of GPUs
pf.rocm_get_device_info(0)      # Detailed GPU info

# Common operations
pf.matmul(a, b)                 # Matrix multiply
pf.conv2d(x, w, padding=1)      # 2D convolution
pf.relu(x)                      # ReLU activation
pf.gelu(x)                      # GELU activation
pf.softmax(x, dim=-1)           # Softmax
pf.cross_entropy(logits, targets)  # Cross-entropy loss
```
