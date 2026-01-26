# AMD ROCm Backend for PyFlame

## Overview

PyFlame supports AMD GPUs through the ROCm backend, enabling high-performance deep learning on AMD Instinct and Radeon GPUs. This backend breaks vendor lock-in by providing an alternative to CUDA/NVIDIA for GPU-accelerated computing.

The ROCm backend uses:
- **rocBLAS** for matrix operations (GEMM, batched GEMM)
- **MIOpen** for DNN operations (convolution, pooling, batch norm, activations)
- **Custom HIP kernels** for operations not covered by libraries (GELU, SiLU, comparisons, loss functions)

## Requirements

### Hardware
- AMD Instinct GPUs: MI100, MI200 (MI210, MI250, MI250X), MI300 series
- AMD Radeon GPUs: RX 6000 series, RX 7000 series (with ROCm support)
- At least 8GB GPU memory recommended

### Software
- ROCm 5.4 or later (6.0+ recommended for best performance)
- Linux operating system (Ubuntu 20.04/22.04, RHEL 8/9, SLES 15)
- CMake 3.21 or later
- Python 3.8+ (for Python bindings)

## Installation

### Installing ROCm

Follow the official ROCm installation guide for your distribution:

```bash
# Ubuntu 22.04 example
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocm-smi
/opt/rocm/bin/rocminfo
```

### Building PyFlame with ROCm Support

```bash
# Clone PyFlame
git clone https://github.com/your-org/pyflame.git
cd pyflame

# Create build directory
mkdir build && cd build

# Configure with ROCm support
cmake -DPYFLAME_USE_ROCM=ON \
      -DROCM_PATH=/opt/rocm \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Build
make -j$(nproc)

# Install Python package
pip install -e ../python
```

### Verifying Installation

```python
import pyflame as pf

# Check ROCm availability
print(f"ROCm available: {pf.rocm_is_available()}")
print(f"GPU count: {pf.rocm_device_count()}")

# Get device info
if pf.rocm_is_available():
    info = pf.rocm_get_device_info(0)
    print(f"GPU name: {info['name']}")
    print(f"Architecture: {info['architecture']}")
    print(f"Total memory: {info['total_memory'] / 1e9:.1f} GB")
    print(f"Compute units: {info['compute_units']}")
```

## Usage

### Selecting the ROCm Backend

```python
import pyflame as pf

# Use first AMD GPU
pf.set_device('rocm')

# Use specific GPU (for multi-GPU systems)
pf.set_device('rocm:0')  # First GPU
pf.set_device('rocm:1')  # Second GPU

# Check current device
print(pf.get_device())  # 'rocm:0'

# Get device information
info = pf.device_info()
print(info)
```

### Running Operations

```python
import pyflame as pf

pf.set_device('rocm')

# Create tensors
a = pf.randn(1024, 1024)
b = pf.randn(1024, 1024)

# Matrix multiplication (uses rocBLAS)
c = pf.matmul(a, b)

# Activation functions
x = pf.randn(10000)
y_relu = pf.relu(x)
y_gelu = pf.gelu(x)
y_silu = pf.silu(x)

# Convolution (uses MIOpen)
input = pf.randn(32, 3, 224, 224)
weight = pf.randn(64, 3, 7, 7)
output = pf.conv2d(input, weight, stride=2, padding=3)
```

### Running Neural Networks

```python
import pyflame as pf
import pyflame.nn as nn

pf.set_device('rocm')

# Define a simple model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Forward pass runs on AMD GPU
x = pf.randn(32, 784)  # Batch of 32
logits = model(x)

# Compute loss
targets = pf.randint(0, 10, (32,))
loss = pf.cross_entropy(logits, targets)
```

### Synchronization

ROCm operations are asynchronous. Use `synchronize()` when you need to wait for completion:

```python
import pyflame as pf
import time

pf.set_device('rocm')

x = pf.randn(10000, 10000)

start = time.time()
for _ in range(100):
    y = pf.matmul(x, x)
pf.synchronize()  # Wait for all operations to complete
elapsed = time.time() - start

print(f"Time: {elapsed:.3f}s")
```

## Supported Operations

All 65 PyFlame operations are supported on the ROCm backend:

### Matrix Operations
| Operation | Implementation |
|-----------|----------------|
| matmul | rocBLAS GEMM |
| batch_matmul | rocBLAS Strided Batched GEMM |
| transpose | rocBLAS geam |

### Convolution Operations
| Operation | Implementation |
|-----------|----------------|
| conv1d | MIOpen Convolution |
| conv2d | MIOpen Convolution |
| conv3d | MIOpen Convolution |
| conv_transpose2d | MIOpen Convolution Backward Data |

### Pooling Operations
| Operation | Implementation |
|-----------|----------------|
| max_pool1d/2d/3d | MIOpen Pooling |
| avg_pool1d/2d/3d | MIOpen Pooling |
| adaptive_avg_pool2d | Custom HIP Kernel |

### Normalization Operations
| Operation | Implementation |
|-----------|----------------|
| batch_norm | MIOpen Batch Normalization |
| layer_norm | MIOpen Layer Normalization |
| group_norm | MIOpen + Custom HIP |

### Activation Functions
| Operation | Implementation |
|-----------|----------------|
| relu | MIOpen Activation |
| sigmoid | MIOpen Activation |
| tanh | MIOpen Activation |
| gelu | Custom HIP Kernel |
| silu | Custom HIP Kernel |
| softmax | MIOpen Softmax |
| log_softmax | MIOpen Softmax |

### Elementwise Operations
| Operation | Implementation |
|-----------|----------------|
| add, sub, mul, div | Custom HIP Kernel |
| pow, max, min | Custom HIP Kernel |
| neg, abs, sqrt, exp, log | Custom HIP Kernel |
| sin, cos, tanh | Custom HIP Kernel |

### Comparison Operations
| Operation | Implementation |
|-----------|----------------|
| eq, ne, lt, le, gt, ge | Custom HIP Kernel |
| where | Custom HIP Kernel |
| clamp | Custom HIP Kernel |

### Reduction Operations
| Operation | Implementation |
|-----------|----------------|
| sum, mean | MIOpen Reduce |
| max, min | MIOpen Reduce |
| prod | Custom HIP Kernel |

### Loss Functions
| Operation | Implementation |
|-----------|----------------|
| mse_loss | Custom HIP Kernel |
| cross_entropy | Custom HIP Kernel |
| bce_loss | Custom HIP Kernel |
| nll_loss | Custom HIP Kernel |

## Performance Optimization

### Enable MIOpen Auto-tuning

MIOpen can find the fastest algorithm for your specific GPU and problem size:

```python
import pyflame as pf
from pyflame.backend.rocm import get_tuning_cache

# Enable exhaustive search (slower first run, faster subsequent runs)
cache = get_tuning_cache()
cache.enable_exhaustive_search(True)

# Optionally persist the cache to disk
cache.set_cache_file('/path/to/miopen_cache.bin')
```

### Use Pinned Memory for Data Loading

Pinned (page-locked) memory enables faster host-to-device transfers:

```python
import pyflame as pf
from pyflame.backend.rocm import get_pinned_pool

# Pre-warm the pinned memory pool
pool = get_pinned_pool()
pool.warmup([batch_size * 3 * 224 * 224 * 4])  # Image batch size
```

### Memory Pool Warmup

Reduce first-run latency by pre-allocating common buffer sizes:

```python
import pyflame as pf
from pyflame.backend.rocm import warmup_memory_pool

# Warmup for a specific batch size and model size
warmup_memory_pool(batch_size=32, model_size='medium')  # 'small', 'medium', 'large'
```

### Kernel Fusion

PyFlame automatically fuses common operation patterns:
- matmul + bias + ReLU
- matmul + bias + GELU
- conv + batch_norm + ReLU
- add + ReLU

### Best Practices

1. **Use appropriate batch sizes**: Larger batches better utilize GPU parallelism
2. **Minimize host-device transfers**: Keep data on GPU as long as possible
3. **Use mixed precision**: FP16/BF16 can significantly improve performance
4. **Profile your workload**: Use `rocprof` or the built-in benchmarks

## Running Benchmarks

PyFlame includes a comprehensive benchmark suite:

```bash
# Run all benchmarks
python benchmarks/rocm_benchmark.py --all

# Run specific benchmarks
python benchmarks/rocm_benchmark.py --matmul --conv

# Custom iterations
python benchmarks/rocm_benchmark.py --iterations 200 --warmup 20
```

## Troubleshooting

### "ROCm not available"

1. Verify ROCm installation:
   ```bash
   rocm-smi
   /opt/rocm/bin/rocminfo
   ```

2. Check that your GPU is supported:
   ```bash
   /opt/rocm/bin/rocminfo | grep "Name:"
   ```

3. Ensure PyFlame was built with ROCm support:
   ```bash
   cmake -L | grep ROCM
   ```

### Out of Memory Errors

1. Check available GPU memory:
   ```bash
   rocm-smi --showmeminfo vram
   ```

2. Reduce batch size or model size

3. Use gradient checkpointing for training

4. Clear memory pool:
   ```python
   from pyflame.backend.rocm import get_memory_manager
   get_memory_manager().clear_pool()
   ```

### Numerical Differences from CPU

Small numerical differences (< 1e-5) between CPU and GPU results are normal due to:
- Different floating-point operation ordering
- GPU-specific math library implementations
- Fused multiply-add operations

For debugging, you can compare with CPU:
```python
pf.set_device('rocm')
result_gpu = pf.matmul(a, b).numpy()

pf.set_device('cpu')
result_cpu = pf.matmul(a, b).numpy()

import numpy as np
np.testing.assert_allclose(result_gpu, result_cpu, rtol=1e-5, atol=1e-6)
```

### Performance Issues

1. Enable MIOpen auto-tuning
2. Check if other processes are using the GPU
3. Verify ROCm version compatibility
4. Use `rocprof` to identify bottlenecks:
   ```bash
   rocprof --stats python your_script.py
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCM_PATH` | ROCm installation directory | `/opt/rocm` |
| `HIP_VISIBLE_DEVICES` | Visible GPU devices | All |
| `MIOPEN_FIND_MODE` | MIOpen algorithm search mode | `NORMAL` |
| `MIOPEN_USER_DB_PATH` | MIOpen kernel cache location | `~/.config/miopen` |

## Multi-GPU Support

```python
import pyflame as pf

# Check available GPUs
num_gpus = pf.rocm_device_count()
print(f"Available GPUs: {num_gpus}")

# Use specific GPU
for i in range(num_gpus):
    pf.set_device(f'rocm:{i}')
    info = pf.device_info()
    print(f"GPU {i}: {info['name']}")
```

## See Also

- [ROCm API Reference](../api/rocm_api.md)
- [Getting Started with ROCm](../tutorials/getting_started_rocm.md)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/)
