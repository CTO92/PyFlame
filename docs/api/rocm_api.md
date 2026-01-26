# ROCm Backend API Reference

This document provides a complete API reference for PyFlame's ROCm backend.

## Python API

### Device Management

#### `rocm_is_available()`

Check if ROCm GPU backend is available.

```python
def rocm_is_available() -> bool
```

**Returns:**
- `bool`: True if ROCm is available and at least one compatible GPU is detected.

**Example:**
```python
import pyflame as pf

if pf.rocm_is_available():
    print("ROCm is available!")
else:
    print("ROCm not available, using CPU")
```

---

#### `rocm_device_count()`

Get the number of available ROCm GPU devices.

```python
def rocm_device_count() -> int
```

**Returns:**
- `int`: Number of available AMD GPUs. Returns 0 if ROCm is not available.

**Example:**
```python
import pyflame as pf

num_gpus = pf.rocm_device_count()
print(f"Found {num_gpus} AMD GPU(s)")
```

---

#### `rocm_get_device_info(device_id=0)`

Get detailed information about a specific ROCm device.

```python
def rocm_get_device_info(device_id: int = 0) -> dict
```

**Parameters:**
- `device_id` (int): Device index (0-based). Default: 0.

**Returns:**
- `dict`: Dictionary containing device information:
  - `device_id` (int): Device index
  - `name` (str): Device name (e.g., "AMD Instinct MI100")
  - `architecture` (str): GPU architecture (e.g., "gfx908")
  - `total_memory` (int): Total GPU memory in bytes
  - `free_memory` (int): Available GPU memory in bytes
  - `compute_units` (int): Number of compute units

**Raises:**
- `RuntimeError`: If device_id is invalid or ROCm is not available.

**Example:**
```python
import pyflame as pf

info = pf.rocm_get_device_info(0)
print(f"GPU: {info['name']}")
print(f"Memory: {info['total_memory'] / 1e9:.1f} GB")
print(f"Free: {info['free_memory'] / 1e9:.1f} GB")
print(f"Compute Units: {info['compute_units']}")
```

---

#### `rocm_set_device(device_id)`

Set the current ROCm device for subsequent operations.

```python
def rocm_set_device(device_id: int) -> None
```

**Parameters:**
- `device_id` (int): Device index to use.

**Raises:**
- `RuntimeError`: If device_id is invalid or ROCm is not available.

**Example:**
```python
import pyflame as pf

# Use second GPU
pf.rocm_set_device(1)
```

---

#### `rocm_get_device()`

Get the current ROCm device index.

```python
def rocm_get_device() -> int
```

**Returns:**
- `int`: Current device index. Returns -1 if no device is set.

---

#### `rocm_synchronize()`

Synchronize the current ROCm device.

Blocks until all previously queued operations on the current device have completed.

```python
def rocm_synchronize() -> None
```

**Example:**
```python
import pyflame as pf
import time

pf.set_device('rocm')

start = time.time()
for _ in range(100):
    y = pf.matmul(x, x)
pf.rocm_synchronize()  # Wait for completion
elapsed = time.time() - start
```

---

### High-Level Device Functions

#### `set_device(device)`

Set the compute device for PyFlame operations.

```python
def set_device(device: str) -> None
```

**Parameters:**
- `device` (str): Device specification string:
  - `'cpu'`: Use CPU backend
  - `'rocm'`: Use first AMD GPU
  - `'rocm:N'`: Use AMD GPU at index N
  - `'cerebras'`: Use Cerebras hardware
  - `'cerebras:simulator'`: Use Cerebras simulator

**Raises:**
- `ValueError`: If device specification is invalid.
- `RuntimeError`: If requested device is not available.

**Example:**
```python
import pyflame as pf

pf.set_device('rocm')      # First AMD GPU
pf.set_device('rocm:1')    # Second AMD GPU
pf.set_device('cpu')       # Switch to CPU
```

---

#### `get_device()`

Get the current compute device.

```python
def get_device() -> str
```

**Returns:**
- `str`: Device specification string (e.g., `'rocm:0'`, `'cpu'`).

---

#### `device_info()`

Get information about the current compute device.

```python
def device_info() -> dict
```

**Returns:**
- `dict`: Device information. Contents depend on device type:
  - For ROCm: Same as `rocm_get_device_info()` plus `type='rocm'`
  - For CPU: `{'type': 'cpu', 'name': ..., 'cores': ...}`

---

#### `synchronize()`

Synchronize the current compute device.

```python
def synchronize() -> None
```

For ROCm devices, waits for all GPU operations to complete.
For CPU, this is a no-op.

---

## C++ API

### Namespace

All ROCm backend functionality is in the `pyflame::backend::rocm` namespace.

### Device Management

```cpp
namespace pyflame::backend::rocm {

/// Check if ROCm is available
bool is_available();

/// Get number of devices
int get_device_count();

/// Get current device
int get_current_device();

/// Set current device
void set_device(int device_id);

/// Synchronize current device
void synchronize();

/// Device information structure
struct DeviceInfo {
    int device_id;
    std::string name;
    std::string architecture;
    size_t total_memory;
    size_t free_memory;
    int compute_units;
};

/// Get device information
DeviceInfo get_device_info(int device_id);

}
```

### Memory Management

```cpp
namespace pyflame::backend::rocm {

/// Memory statistics
struct MemoryStats {
    size_t total_allocated;
    size_t peak_allocated;
    size_t total_allocations;
    size_t total_deallocations;
    size_t cache_hits;
    size_t cache_misses;
};

/// GPU Memory Manager
class ROCmMemoryManager {
public:
    struct Config {
        bool enable_pooling = true;
        size_t pool_size_mb = 256;
        size_t max_pool_size_mb = 4096;
        bool use_pinned_host = true;
    };

    explicit ROCmMemoryManager(Config config = {});
    ~ROCmMemoryManager();

    // GPU memory
    void* allocate(size_t bytes);
    void* allocate_async(size_t bytes, hipStream_t stream);
    void deallocate(void* ptr);
    void deallocate_async(void* ptr, hipStream_t stream);

    // Memory transfers
    void copy_host_to_device(void* dst, const void* src, size_t bytes);
    void copy_device_to_host(void* dst, const void* src, size_t bytes);
    void copy_device_to_device(void* dst, const void* src, size_t bytes);

    // Async transfers
    void copy_host_to_device_async(void* dst, const void* src, size_t bytes, hipStream_t stream);
    void copy_device_to_host_async(void* dst, const void* src, size_t bytes, hipStream_t stream);
    void copy_device_to_device_async(void* dst, const void* src, size_t bytes, hipStream_t stream);

    // Memory operations
    void memset(void* ptr, int value, size_t bytes);
    void memset_async(void* ptr, int value, size_t bytes, hipStream_t stream);

    // Pinned host memory
    void* allocate_pinned(size_t bytes);
    void deallocate_pinned(void* ptr);

    // Statistics
    MemoryStats get_stats() const;
    void reset_stats();
    void clear_pool();
    void get_memory_info(size_t* free, size_t* total) const;
};

/// Global memory manager singleton
ROCmMemoryManager& get_memory_manager();

}
```

### rocBLAS Wrapper

```cpp
namespace pyflame::backend::rocm {

/// rocBLAS wrapper for matrix operations
class ROCmBLAS {
public:
    ROCmBLAS();
    ~ROCmBLAS();

    /// Set the HIP stream
    void set_stream(hipStream_t stream);

    /// Get rocBLAS handle
    rocblas_handle handle() const;

    /// General matrix multiply (GEMM)
    /// C = alpha * op(A) * op(B) + beta * C
    void gemm(
        rocblas_operation trans_a,
        rocblas_operation trans_b,
        int M, int N, int K,
        float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta,
        float* C, int ldc
    );

    /// Strided batched GEMM
    void gemm_strided_batched(
        rocblas_operation trans_a,
        rocblas_operation trans_b,
        int M, int N, int K,
        float alpha,
        const float* A, int lda, int64_t stride_a,
        const float* B, int ldb, int64_t stride_b,
        float beta,
        float* C, int ldc, int64_t stride_c,
        int batch_count
    );

    /// Matrix addition/scaling (geam)
    void geam(
        rocblas_operation trans_a,
        rocblas_operation trans_b,
        int M, int N,
        float alpha,
        const float* A, int lda,
        float beta,
        const float* B, int ldb,
        float* C, int ldc
    );
};

/// Global BLAS singleton
ROCmBLAS& get_blas();

}
```

### MIOpen Wrapper

```cpp
namespace pyflame::backend::rocm {

/// Convolution parameters
struct ConvParams {
    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
    int groups = 1;

    static ConvParams Conv2D(int pad_h, int pad_w,
                             int stride_h = 1, int stride_w = 1,
                             int dilation_h = 1, int dilation_w = 1,
                             int groups = 1);
};

/// Pooling parameters
struct PoolParams {
    std::vector<int> kernel_size;
    std::vector<int> padding;
    std::vector<int> stride;
    bool ceil_mode = false;

    static PoolParams Pool2D(int kernel_h, int kernel_w,
                             int pad_h = 0, int pad_w = 0,
                             int stride_h = 1, int stride_w = 1);
};

/// Batch normalization parameters
struct BatchNormParams {
    double epsilon = 1e-5;
    double momentum = 0.1;
    bool training = true;
};

/// MIOpen wrapper for DNN operations
class ROCmDNN {
public:
    ROCmDNN();
    ~ROCmDNN();

    void set_stream(hipStream_t stream);
    miopenHandle_t handle() const;

    // Convolution
    void conv2d_forward(
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const void* input,
        const std::vector<int64_t>& weight_shape,
        const void* weight,
        const std::vector<int64_t>& output_shape,
        void* output,
        const ConvParams& params,
        void* workspace,
        size_t workspace_size
    );

    size_t conv2d_forward_workspace_size(
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const std::vector<int64_t>& weight_shape,
        const std::vector<int64_t>& output_shape,
        const ConvParams& params
    );

    // Pooling
    void pooling2d_forward(
        ir::OpType pool_type,
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const void* input,
        const std::vector<int64_t>& output_shape,
        void* output,
        const PoolParams& params
    );

    // Batch Normalization
    void batchnorm_forward_training(...);
    void batchnorm_forward_inference(...);

    // Activations
    void activation_forward(
        ir::OpType activation_type,
        DType dtype,
        const std::vector<int64_t>& shape,
        const void* input,
        void* output,
        float alpha = 0.0f
    );

    // Softmax
    void softmax_forward(
        DType dtype,
        const std::vector<int64_t>& shape,
        const void* input,
        void* output,
        int axis = -1,
        bool log_softmax = false
    );

    // Reductions
    void reduce(
        ir::OpType reduce_type,
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const void* input,
        const std::vector<int64_t>& output_shape,
        void* output,
        const std::vector<int>& reduce_dims,
        void* workspace,
        size_t workspace_size
    );
};

/// Global DNN singleton
ROCmDNN& get_dnn();

}
```

### Custom HIP Kernels

```cpp
namespace pyflame::backend::rocm {

// Activation kernels
void launch_gelu_forward(float* output, const float* input, int64_t numel, hipStream_t stream);
void launch_silu_forward(float* output, const float* input, int64_t numel, hipStream_t stream);

// Elementwise binary operations
void launch_elementwise_binary(
    ir::OpType op,
    float* output,
    const float* a, const float* b,
    int64_t numel, hipStream_t stream
);

// Elementwise unary operations
void launch_elementwise_unary(
    ir::OpType op,
    float* output, const float* input,
    int64_t numel, hipStream_t stream
);

// Comparison operations
void launch_comparison(
    ir::OpType op,
    float* output,
    const float* a, const float* b,
    int64_t numel, hipStream_t stream
);

// Conditional operations
void launch_where(
    float* output,
    const float* condition,
    const float* x, const float* y,
    int64_t numel, hipStream_t stream
);

void launch_clamp(
    float* output, const float* input,
    float min_val, float max_val,
    int64_t numel, hipStream_t stream
);

// Loss functions
void launch_mse_loss(
    float* output,
    const float* predictions, const float* targets,
    int64_t numel, hipStream_t stream
);

void launch_bce_loss(
    float* output,
    const float* predictions, const float* targets,
    int64_t numel, hipStream_t stream
);

void launch_cross_entropy_loss(
    float* output,
    const float* logits, const int32_t* targets,
    int64_t N, int64_t C, hipStream_t stream
);

}
```

### Graph Executor

```cpp
namespace pyflame::backend::rocm {

/// ROCm graph executor
class ROCmExecutor {
public:
    struct Config {
        int device_id = 0;
        bool enable_profiling = false;
        bool sync_after_launch = false;
        size_t workspace_size_mb = 256;
    };

    explicit ROCmExecutor(Config config = {});
    ~ROCmExecutor();

    /// Execute a computation graph
    ExecutionResult execute(
        ir::Graph& graph,
        const std::vector<ir::NodeId>& output_ids
    );

    /// Get device info
    DeviceInfo get_device_info() const;

    /// Get memory statistics
    MemoryStats get_memory_stats() const;
};

}
```

### Performance Optimization

```cpp
namespace pyflame::backend::rocm {

/// MIOpen tuning cache
class MIOpenTuningCache {
public:
    struct Config {
        bool enable_exhaustive_search = true;
        bool enable_caching = true;
        bool enable_persistence = false;
        std::string cache_file;
    };

    explicit MIOpenTuningCache(Config config = {});

    void enable_exhaustive_search(bool enable);
    void set_cache_file(const std::string& path);
    void clear();

    struct Stats {
        size_t hits;
        size_t misses;
        size_t entries;
    };
    Stats get_stats() const;
};

/// Global tuning cache
MIOpenTuningCache& get_tuning_cache();

/// Pinned memory pool
class PinnedMemoryPool {
public:
    void* acquire(size_t bytes);
    void release(void* ptr);
    void warmup(const std::vector<size_t>& sizes);
    void clear();
};

/// Global pinned pool
PinnedMemoryPool& get_pinned_pool();

/// Memory warmup
void warmup_memory_pool(int batch_size, const std::string& model_size);

/// Kernel fusion
enum class FusedOpType {
    MATMUL_BIAS,
    MATMUL_BIAS_RELU,
    MATMUL_BIAS_GELU,
    CONV_BIAS,
    CONV_BIAS_RELU,
    CONV_BN_RELU,
    ADD_RELU,
    MUL_ADD
};

std::optional<FusedOpType> can_fuse(const std::vector<ir::OpType>& ops);

}
```

## Error Handling

### HIP Errors

The ROCm backend uses the `HIP_CHECK` macro for HIP API calls:

```cpp
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error( \
            std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)
```

### MIOpen Errors

```cpp
#define MIOPEN_CHECK(call) do { \
    miopenStatus_t status = call; \
    if (status != miopenStatusSuccess) { \
        throw std::runtime_error("MIOpen error: " + std::to_string(status)); \
    } \
} while(0)
```

### rocBLAS Errors

```cpp
#define ROCBLAS_CHECK(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        throw std::runtime_error("rocBLAS error: " + std::to_string(status)); \
    } \
} while(0)
```

## Thread Safety

- `ROCmMemoryManager`: Thread-safe (uses internal mutex)
- `ROCmBLAS`: Single-threaded (one handle per instance)
- `ROCmDNN`: Single-threaded (one handle per instance)
- `MIOpenTuningCache`: Thread-safe (uses internal mutex)
- `PinnedMemoryPool`: Thread-safe (uses internal mutex)

For multi-threaded applications, create separate executor instances per thread or use appropriate synchronization.

## See Also

- [ROCm Backend User Guide](../backends/rocm.md)
- [Getting Started with ROCm](../tutorials/getting_started_rocm.md)
