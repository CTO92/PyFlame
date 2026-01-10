# PyFlame Integration Guide

**PyFlame Version:** Pre-Release Alpha 1.0

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs described here are subject to change.

---

## Overview

This guide explains how to integrate PyFlame into your existing projects, whether you're working with C++, Python, or both.

---

## Python Integration

### Method 1: Development Installation (Recommended)

For active development or when you need the latest features:

```bash
# Clone and build PyFlame
git clone https://github.com/CTO92/PyFlame.git
cd PyFlame

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Install in development mode
cd ..
pip install -e .
```

This creates a symlink to the source, allowing you to modify PyFlame and see changes immediately.

### Method 2: System Installation

For production or stable usage:

```bash
# After building
cd PyFlame
pip install .
```

### Using PyFlame in Your Python Project

```python
import pyflame as pf

# Your existing model code
class MyModel:
    def __init__(self):
        self.weights = pf.randn([784, 256]) * 0.01
        self.bias = pf.zeros([256])

    def forward(self, x):
        return pf.relu(x @ self.weights + self.bias)

# Use with existing data pipeline
model = MyModel()
input_data = pf.from_numpy(your_numpy_data)
output = model.forward(input_data)
pf.eval(output)
```

### Coexistence with NumPy and PyTorch

PyFlame is designed to work alongside existing libraries:

```python
import numpy as np
import pyflame as pf

# NumPy preprocessing
data = np.load('data.npy')
data = (data - data.mean()) / data.std()  # Normalize with NumPy

# PyFlame computation
tensor = pf.from_numpy(data.astype(np.float32))
result = pf.relu(tensor @ weights)
pf.eval(result)

# Back to NumPy for visualization/saving
output = result.numpy()
np.save('output.npy', output)
```

---

## C++ Integration

### Method 1: CMake FetchContent (Recommended)

Add PyFlame directly to your CMake project:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MyProject)

include(FetchContent)

FetchContent_Declare(
    pyflame
    GIT_REPOSITORY https://github.com/CTO92/PyFlame.git
    GIT_TAG main  # or specific version tag
)
FetchContent_MakeAvailable(pyflame)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE pyflame::pyflame)
```

### Method 2: Subdirectory

If you've cloned PyFlame into your project:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MyProject)

add_subdirectory(external/PyFlame)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE pyflame::pyflame)
```

### Method 3: Installed Package

If PyFlame is installed system-wide:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MyProject)

find_package(pyflame REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE pyflame::pyflame)
```

To install PyFlame system-wide:

```bash
cd PyFlame/build
cmake --install . --prefix /usr/local
```

### Using PyFlame in Your C++ Code

```cpp
#include <pyflame/pyflame.hpp>

using namespace pyflame;

int main() {
    // Create tensors
    auto input = Tensor::randn({32, 784});
    auto weights = Tensor::randn({784, 256}) * 0.01f;
    auto bias = Tensor::zeros({256});

    // Build computation
    auto hidden = relu(matmul(input, weights) + bias);
    auto output = hidden.sum();

    // Execute
    output.eval();

    std::cout << "Result: " << output.data<float>()[0] << std::endl;

    return 0;
}
```

---

## Build Configuration Options

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `PYFLAME_BUILD_TESTS` | ON | Build unit tests |
| `PYFLAME_BUILD_EXAMPLES` | ON | Build example programs |
| `PYFLAME_BUILD_PYTHON` | ON | Build Python bindings |
| `PYFLAME_USE_CEREBRAS_SDK` | OFF | Enable Cerebras SDK integration |
| `CEREBRAS_SDK_PATH` | "" | Path to Cerebras SDK (if enabled) |

### Example: Minimal Build

For a smaller build without tests or examples:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYFLAME_BUILD_TESTS=OFF \
    -DPYFLAME_BUILD_EXAMPLES=OFF
```

### Example: With Cerebras SDK

For WSE hardware execution:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYFLAME_USE_CEREBRAS_SDK=ON \
    -DCEREBRAS_SDK_PATH=/opt/cerebras/sdk
```

---

## Project Structure Recommendations

### Python Project

```
my_ml_project/
├── pyproject.toml
├── setup.py
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── model.py      # Uses pyflame
│       └── train.py
├── tests/
│   └── test_model.py
└── requirements.txt      # Include pyflame path or git URL
```

**requirements.txt:**
```
numpy>=1.20
pyflame @ git+https://github.com/CTO92/PyFlame.git
```

### C++ Project

```
my_cpp_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   └── model.cpp
├── include/
│   └── model.hpp
├── external/
│   └── PyFlame/          # Git submodule or FetchContent
└── tests/
    └── test_model.cpp
```

---

## Cerebras WSE Configuration

### Environment Setup

For on-premises Cerebras systems:

```bash
# Set runtime address
export CEREBRAS_RUNTIME_ADDRESS="localhost:9000"

# Or for specific IP
export CEREBRAS_RUNTIME_ADDRESS="192.168.1.100:9000"
```

For Cerebras Cloud:

```bash
# Use endpoint from your cloud instance
export CEREBRAS_RUNTIME_ADDRESS="your-instance.cloud.cerebras.net:port"
```

### Programmatic Configuration

**Python:**
```python
import pyflame as pf

options = pf.CodeGenOptions()
options.target = "wse2"  # or "wse3"
options.runtime_address = "localhost:9000"
options.optimize = True

# Generate CSL code
generator = pf.CSLCodeGenerator()
result = generator.generate(graph, options)
```

**C++:**
```cpp
#include <pyflame/backend/csl_codegen.hpp>

pyflame::backend::CodeGenOptions options;
options.target = "wse2";
options.runtime_address = "localhost:9000";
options.optimize = true;

pyflame::backend::CSLCodeGenerator generator;
auto result = generator.generate(graph, options);
```

---

## Handling Data Types

### Type Compatibility

| PyFlame Type | NumPy Type | C++ Type |
|--------------|------------|----------|
| `float32` | `np.float32` | `float` |
| `float16` | `np.float16` | `half` (if available) |
| `bfloat16` | N/A | `bfloat16` |
| `int32` | `np.int32` | `int32_t` |
| `int16` | `np.int16` | `int16_t` |
| `int8` | `np.int8` | `int8_t` |
| `bool_` | `np.bool_` | `bool` |

### Type Conversion Best Practices

```python
import numpy as np
import pyflame as pf

# Ensure float32 for best compatibility
data = np.asarray(your_data, dtype=np.float32)
tensor = pf.from_numpy(data)

# Specify type explicitly when needed
x = pf.zeros([100, 100], dtype=pf.float16)  # Half precision for memory savings
```

---

## Memory Management

### Python

PyFlame tensors are automatically garbage collected. However, for large computations:

```python
import pyflame as pf

def process_batch(batch):
    # Tensors created here are cleaned up after function returns
    x = pf.from_numpy(batch)
    result = pf.relu(x @ weights)
    pf.eval(result)
    return result.numpy()  # Return NumPy, allowing PyFlame tensor to be freed
```

### C++

PyFlame uses shared pointers internally. Tensors are freed when all references go out of scope:

```cpp
void process() {
    auto x = Tensor::randn({1000, 1000});
    auto y = relu(x);
    y.eval();
    // Both x and y cleaned up when function returns
}
```

For explicit control:

```cpp
{
    auto large_tensor = Tensor::randn({10000, 10000});
    // Use tensor...
}  // Tensor freed here
```

---

## Error Handling

### Python

```python
import pyflame as pf

try:
    # Shape mismatch will raise exception
    a = pf.randn([100, 50])
    b = pf.randn([60, 75])  # Wrong shape for matmul
    c = a @ b
except RuntimeError as e:
    print(f"PyFlame error: {e}")
```

### C++

```cpp
#include <pyflame/pyflame.hpp>
#include <stdexcept>

try {
    auto a = Tensor::randn({100, 50});
    auto b = Tensor::randn({60, 75});
    auto c = matmul(a, b);  // Will throw
} catch (const std::runtime_error& e) {
    std::cerr << "PyFlame error: " << e.what() << std::endl;
}
```

---

## Testing Your Integration

### Python Tests with pytest

```python
# test_integration.py
import pytest
import numpy as np
import pyflame as pf

def test_tensor_creation():
    x = pf.zeros([3, 4])
    assert x.shape == [3, 4]

def test_numpy_roundtrip():
    original = np.random.randn(10, 10).astype(np.float32)
    tensor = pf.from_numpy(original)
    pf.eval(tensor)
    recovered = tensor.numpy()
    np.testing.assert_allclose(original, recovered, rtol=1e-5)

def test_computation():
    a = pf.randn([32, 64])
    b = pf.randn([64, 32])
    c = a @ b
    pf.eval(c)
    assert c.shape == [32, 32]
```

### C++ Tests with Google Test

```cpp
#include <gtest/gtest.h>
#include <pyflame/pyflame.hpp>

using namespace pyflame;

TEST(Integration, TensorCreation) {
    auto x = Tensor::zeros({3, 4});
    EXPECT_EQ(x.shape()[0], 3);
    EXPECT_EQ(x.shape()[1], 4);
}

TEST(Integration, MatMul) {
    auto a = Tensor::randn({32, 64});
    auto b = Tensor::randn({64, 32});
    auto c = matmul(a, b);
    c.eval();
    EXPECT_EQ(c.shape()[0], 32);
    EXPECT_EQ(c.shape()[1], 32);
}
```

---

## Troubleshooting

### Common Issues

**Import Error: Module not found**
```
ModuleNotFoundError: No module named 'pyflame._pyflame_cpp'
```
Solution: Ensure you've built the C++ extension and installed the package:
```bash
cd PyFlame/build && cmake --build . --config Release
cd .. && pip install -e .
```

**Link Error: Undefined symbols**
```
undefined reference to `pyflame::Tensor::zeros'
```
Solution: Ensure you're linking against pyflame in CMake:
```cmake
target_link_libraries(your_target PRIVATE pyflame::pyflame)
```

**Shape Mismatch Errors**
```
RuntimeError: Shape mismatch in matmul: [100, 50] @ [60, 75]
```
Solution: Check tensor dimensions. For matmul, inner dimensions must match.

### Debug Mode

Build with debug info for better error messages:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DPYFLAME_BUILD_TESTS=ON
```

---

## Next Steps

- See [Getting Started](getting_started.md) for a tutorial introduction
- Check [API Reference](api_reference.md) for complete documentation
- Review [Examples](examples.md) for more integration patterns
- Read [Best Practices](best_practices.md) for optimization tips
