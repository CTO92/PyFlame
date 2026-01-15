# PyFlame

**Native Deep Learning Framework for Cerebras WSE**

> **PRE-RELEASE ALPHA 1.0**
>
> This software is in early development and is not yet ready for production use.
> APIs may change without notice. Use at your own risk.

PyFlame is a tensor computation library designed natively for the Cerebras Wafer-Scale Engine (WSE), featuring lazy evaluation, automatic CSL code generation, and a Python-first API.

## About OA Quantum Labs

PyFlame is developed by **[OA Quantum Labs](https://oaqlabs.com)**, a specialized engineering firm focused on high-performance computing and tooling that breaks vendor lock-in.

### What We Do

In this context We help organizations unlock the full potential of specialized hardware through custom developer tools, optimized frameworks, and performance engineering:

- **Custom Framework Development** — Native tooling designed for your specific accelerator architecture
- **Performance Optimization** — Squeeze maximum throughput from your existing hardware investments
- **Migration & Porting** — Adapt existing ML workloads to new accelerator platforms
- **Training & Enablement** — Get your team productive on specialized hardware faster

### Why Work With Us

PyFlame demonstrates our approach: rather than forcing general-purpose tools onto specialized hardware, we build native solutions that leverage the unique strengths of each architecture. The result is dramatically better performance and a more intuitive developer experience.

If your organization is working with specialized AI accelerators, FPGAs, or custom silicon, we'd love to discuss how purpose-built tooling could transform your development workflow.

### Get In Touch

**Danny Wall** — CTO, OA Quantum Labs
[dwall@oaqlabs.com](mailto:dwall@oaqlabs.com) | [oaqlabs.com](https://oaqlabs.com)

## Features

- **Native WSE Design**: Built from the ground up for Cerebras architecture
- **Lazy Evaluation**: Computation graphs are built lazily and executed on demand
- **CSL Code Generation**: Automatic generation of optimized CSL kernels
- **2D Mesh Layouts**: First-class support for tensor distribution across PEs
- **Python + C++ API**: Use from Python or C++ with the same abstractions
- **NumPy Interoperability**: Easy conversion to/from NumPy arrays
- **PyTorch-like API**: Familiar nn.Module system for building models
- **Automatic Differentiation**: Full autograd support for training
- **Complete Training Stack**: Optimizers, loss functions, and LR schedulers

## Project Status

**Version:** Pre-Release Alpha 1.0

**Phase 1 (Core Infrastructure)** - Complete

- [x] Core tensor class with lazy evaluation
- [x] Computation graph (IR) system
- [x] Shape inference
- [x] Elementwise operations (add, mul, relu, sigmoid, etc.)
- [x] Reduction operations (sum, mean, max, min)
- [x] Matrix multiplication
- [x] CSL code generation framework
- [x] Python bindings via pybind11
- [x] CPU reference implementation

**Phase 2 (ML Primitives)** - Complete

- [x] Automatic differentiation (autograd)
- [x] Neural network module system (nn.Module)
- [x] Linear layers (Linear)
- [x] Convolutional layers (Conv1d, Conv2d)
- [x] Normalization layers (BatchNorm, LayerNorm, GroupNorm)
- [x] Pooling layers (MaxPool, AvgPool, AdaptivePool)
- [x] Dropout layers
- [x] Multi-head attention
- [x] Loss functions (MSE, CrossEntropy, BCE, etc.)
- [x] Optimizers (SGD, Adam, AdamW, RMSprop)
- [x] Learning rate schedulers

## Requirements

- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Python 3.8+
- pybind11 2.10+

### Cerebras SDK (Optional)

PyFlame includes a **CPU reference implementation** that allows you to develop, test, and validate your models without access to Cerebras hardware. All tensor operations, graph building, and CSL code generation work without the SDK.

To actually **execute on Cerebras WSE hardware**, you need:

- **Cerebras SDK** - This is proprietary software available only to Cerebras customers and partners. It is not publicly downloadable.
- **Access to Cerebras hardware** - Either on-premises CS-2/CS-3 systems or Cerebras Cloud.

**Supported deployment options:**
| Environment | Runtime Address | Notes |
|-------------|-----------------|-------|
| On-premises CS-2/CS-3 | `localhost:9000` or system IP | Direct hardware access |
| Cerebras Cloud | Cloud endpoint URL | Provided by your cloud instance |

If you are interested in running PyFlame on Cerebras hardware, please contact [Cerebras Systems](https://www.cerebras.net/contact/) to inquire about SDK access and hardware availability.

To build with Cerebras SDK support (once you have access):
```bash
cmake .. -DPYFLAME_USE_CEREBRAS_SDK=ON -DCEREBRAS_SDK_PATH=/path/to/sdk
```

To configure the runtime endpoint (for cloud or remote on-premises):
```bash
export CEREBRAS_RUNTIME_ADDRESS="your-endpoint:port"
```

## Building

### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/CTO92/PyFlame.git
cd pyflame

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure

# Install Python package (development mode)
pip install -e .
```

### Windows

```powershell
# Clone and build
git clone https://github.com/CTO92/PyFlame.git
cd pyflame

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

ctest -C Release --output-on-failure
```

## Quick Start

### Python

```python
import pyflame as pf

# Create tensors
a = pf.randn([1024, 512])
b = pf.randn([512, 256])

# Build computation graph (lazy)
c = a @ b              # Matrix multiply
d = pf.relu(c)         # Activation
e = d.sum()            # Reduction

# Execute
result = pf.eval(e)
print(result.numpy())

# With explicit mesh layout for WSE
x = pf.zeros([4096, 4096], layout=pf.MeshLayout.grid(16, 16))
y = pf.zeros([4096, 4096], layout=pf.MeshLayout.grid(16, 16))
z = x @ y  # Distributed across 256 PEs
```

### Training a Neural Network

```python
import pyflame as pf
from pyflame import nn, optim

# Define a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Setup optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training step
x = pf.randn([32, 784])  # Batch of inputs
y = pf.randint(0, 10, [32])  # Labels

optimizer.zero_grad()
output = model(x)
loss = loss_fn(output, y)
loss.backward()
optimizer.step()
```

### C++

```cpp
#include <pyflame/pyflame.hpp>
using namespace pyflame;

int main() {
    auto a = Tensor::randn({1024, 512});
    auto b = Tensor::randn({512, 256});

    auto c = matmul(a, b);
    auto d = relu(c);
    auto e = d.sum();

    e.eval();
    std::cout << "Result: " << e.data<float>()[0] << "\n";

    return 0;
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PyFlame User API (Python/C++)            │
│   - Tensor abstraction with lazy evaluation                │
│   - Dataflow-aware operators                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               PyFlame Intermediate Representation           │
│   - Computation graph with shape inference                 │
│   - Optimization passes (fusion, layout, etc.)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PyFlame CSL Backend                        │
│   - Template-based code generation                         │
│   - PE placement and routing optimization                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              CSL Runtime / Cerebras Hardware                │
│   - 850,000+ Processing Elements                           │
│   - 2D mesh fabric with wavelet communication              │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
pyflame/
├── CMakeLists.txt           # Build configuration
├── include/pyflame/         # C++ headers
│   ├── core/                # Tensor, DType, Layout
│   ├── ir/                  # Graph IR, operations
│   └── backend/             # CSL code generation
├── src/                     # C++ implementation
├── python/                  # Python bindings
│   ├── pyflame/             # Python package
│   └── bindings.cpp         # pybind11 bindings
├── tests/                   # Unit tests
│   ├── cpp/                 # C++ tests (Google Test)
│   └── python/              # Python tests (pytest)
├── examples/                # Example programs
│   ├── cpp/
│   └── python/
└── docs/                    # Design documentation
```

## Documentation

### Developer Guides

New to PyFlame? Start here:

- [Getting Started](docs/getting_started.md) - Installation and first steps
- [Integration Guide](docs/integration_guide.md) - Adding PyFlame to your project
- [API Reference](docs/api_reference.md) - Complete function documentation
- [Examples](docs/examples.md) - Practical code examples
- [Best Practices](docs/best_practices.md) - Optimization tips and patterns

### Design Documents

Internal architecture documentation:

- [CSL Code Generation Strategy](docs/01_csl_code_generation.md)
- [Lazy Evaluation & Graph Building](docs/02_lazy_evaluation_graph_building.md)
- [Memory Management & Layouts](docs/03_memory_management_layout.md)
- [Build System Setup](docs/04_build_system.md)
- [Phase 2: ML Primitives Plan](docs/05_phase2_ml_primitives.md)

## API Reference

### Tensor Creation

| Function | Description |
|----------|-------------|
| `pf.zeros(shape)` | Create tensor filled with zeros |
| `pf.ones(shape)` | Create tensor filled with ones |
| `pf.full(shape, value)` | Create tensor filled with value |
| `pf.randn(shape)` | Random normal distribution |
| `pf.rand(shape)` | Random uniform [0, 1) |
| `pf.arange(start, end)` | Range of values |
| `pf.from_numpy(arr)` | From NumPy array |

### Operations

| Category | Functions |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `@` (matmul) |
| Activations | `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softmax` |
| Math | `abs`, `sqrt`, `exp`, `log`, `sin`, `cos` |
| Reductions | `sum`, `mean`, `max`, `min` |
| Shape | `reshape`, `transpose`, `squeeze`, `unsqueeze` |
| Combination | `cat`, `stack` |

### Neural Network Layers (nn)

| Layer | Description |
|-------|-------------|
| `nn.Linear` | Fully connected layer |
| `nn.Conv1d`, `nn.Conv2d` | Convolutional layers |
| `nn.BatchNorm1d`, `nn.BatchNorm2d` | Batch normalization |
| `nn.LayerNorm`, `nn.GroupNorm` | Layer and group normalization |
| `nn.MaxPool2d`, `nn.AvgPool2d` | Pooling layers |
| `nn.Dropout` | Dropout regularization |
| `nn.MultiheadAttention` | Multi-head attention |

### Loss Functions (nn)

| Loss | Description |
|------|-------------|
| `nn.MSELoss` | Mean squared error |
| `nn.L1Loss` | Mean absolute error |
| `nn.CrossEntropyLoss` | Cross-entropy for classification |
| `nn.BCELoss`, `nn.BCEWithLogitsLoss` | Binary cross-entropy |
| `nn.NLLLoss` | Negative log likelihood |
| `nn.KLDivLoss` | KL divergence |

### Optimizers (optim)

| Optimizer | Description |
|-----------|-------------|
| `optim.SGD` | Stochastic gradient descent with momentum |
| `optim.Adam` | Adam optimizer |
| `optim.AdamW` | Adam with decoupled weight decay |
| `optim.RMSprop` | RMSprop optimizer |

### Learning Rate Schedulers (optim)

| Scheduler | Description |
|-----------|-------------|
| `optim.StepLR` | Decay LR every N steps |
| `optim.CosineAnnealingLR` | Cosine annealing schedule |
| `optim.ReduceLROnPlateau` | Reduce LR when metric plateaus |
| `optim.OneCycleLR` | One-cycle learning rate policy |

### Layouts

| Layout | Description |
|--------|-------------|
| `MeshLayout.single_pe()` | All data on one PE |
| `MeshLayout.row_partition(n)` | Split rows across n PEs |
| `MeshLayout.col_partition(n)` | Split columns across n PEs |
| `MeshLayout.grid(r, c)` | 2D tiling across r×c PEs |

## Contributing

Contributions are welcome! Please see our contributing guidelines (coming soon).

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

- Cerebras Systems for the WSE architecture and SDK
- The PyTorch team for API inspiration
- The JAX/XLA team for compiler architecture insights
