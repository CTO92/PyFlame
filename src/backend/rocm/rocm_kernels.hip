// ROCm Custom HIP Kernels Implementation
// Phase 5: Custom HIP Kernels for operations not covered by MIOpen

#ifdef PYFLAME_HAS_ROCM

#include <hip/hip_runtime.h>
#include "pyflame/backend/rocm/rocm_kernels.hpp"
#include "pyflame/backend/rocm/rocm_backend.hpp"
#include <cmath>
#include <cfloat>

namespace pyflame::backend::rocm::kernels {

// ============================================================================
// Constants
// ============================================================================
constexpr int BLOCK_SIZE = 256;
constexpr int64_t MAX_GRID_SIZE = 2147483647LL;  // INT_MAX - safe limit for grid dimension

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate grid size for a given number of elements with overflow protection
/// @throws std::overflow_error if tensor is too large for single kernel launch
inline int get_grid_size_safe(int64_t numel) {
    if (numel <= 0) return 0;
    int64_t blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > MAX_GRID_SIZE) {
        throw std::overflow_error(
            "Tensor size exceeds maximum grid dimension. "
            "numel=" + std::to_string(numel) + " requires " +
            std::to_string(blocks) + " blocks, max is " +
            std::to_string(MAX_GRID_SIZE));
    }
    return static_cast<int>(blocks);
}

/// Legacy function - use get_grid_size_safe for new code
inline int get_grid_size(int64_t numel) {
    return get_grid_size_safe(numel);
}

// ============================================================================
// GELU Kernel
// ============================================================================

__global__ void gelu_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // sqrt(2/pi) ~= 0.7978845608
        float x3 = x * x * x;
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
    }
}

void launch_gelu_forward(
    float* output, const float* input,
    int64_t numel, hipStream_t stream
) {
    if (numel == 0) return;
    int grid = get_grid_size(numel);
    hipLaunchKernelGGL(gelu_forward_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, input, numel);
}

// ============================================================================
// SiLU Kernel
// ============================================================================

__global__ void silu_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        output[idx] = x / (1.0f + expf(-x));
    }
}

void launch_silu_forward(
    float* output, const float* input,
    int64_t numel, hipStream_t stream
) {
    if (numel == 0) return;
    int grid = get_grid_size(numel);
    hipLaunchKernelGGL(silu_forward_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, input, numel);
}

// ============================================================================
// Elementwise Binary Operations
// ============================================================================

// Operation functors for binary operations
struct AddOp { __device__ float operator()(float a, float b) { return a + b; } };
struct SubOp { __device__ float operator()(float a, float b) { return a - b; } };
struct MulOp { __device__ float operator()(float a, float b) { return a * b; } };
// Safe division: returns 0 for 0/0, infinity with correct sign otherwise
struct DivOp {
    __device__ float operator()(float a, float b) {
        if (b == 0.0f) {
            if (a == 0.0f) return 0.0f;  // 0/0 -> 0 (avoid NaN)
            return copysignf(HUGE_VALF, a * b);  // Signed infinity
        }
        return a / b;
    }
};
struct PowOp { __device__ float operator()(float a, float b) { return powf(a, b); } };
struct MaxOp { __device__ float operator()(float a, float b) { return fmaxf(a, b); } };
struct MinOp { __device__ float operator()(float a, float b) { return fminf(a, b); } };

template<typename Op>
__global__ void elementwise_binary_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = Op()(a[idx], b[idx]);
    }
}

void launch_elementwise_binary(
    ir::OpType op,
    float* output,
    const float* a, const float* b,
    int64_t numel, hipStream_t stream
) {
    if (numel == 0) return;
    int grid = get_grid_size(numel);

    switch (op) {
        case ir::OpType::ADD:
            hipLaunchKernelGGL((elementwise_binary_kernel<AddOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::SUB:
            hipLaunchKernelGGL((elementwise_binary_kernel<SubOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::MUL:
            hipLaunchKernelGGL((elementwise_binary_kernel<MulOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::DIV:
            hipLaunchKernelGGL((elementwise_binary_kernel<DivOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::POW:
            hipLaunchKernelGGL((elementwise_binary_kernel<PowOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::MAX_BINARY:
        case ir::OpType::MAXIMUM:
            hipLaunchKernelGGL((elementwise_binary_kernel<MaxOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::MIN_BINARY:
        case ir::OpType::MINIMUM:
            hipLaunchKernelGGL((elementwise_binary_kernel<MinOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        default:
            throw std::runtime_error("Unsupported binary op in kernel");
    }
}

// ============================================================================
// Elementwise Unary Operations
// ============================================================================

// Operation functors for unary operations
struct NegOp  { __device__ float operator()(float x) { return -x; } };
struct AbsOp  { __device__ float operator()(float x) { return fabsf(x); } };
struct SqrtOp { __device__ float operator()(float x) { return sqrtf(x); } };
struct ExpOp  { __device__ float operator()(float x) { return expf(x); } };
struct LogOp  { __device__ float operator()(float x) { return logf(x); } };
struct SinOp  { __device__ float operator()(float x) { return sinf(x); } };
struct CosOp  { __device__ float operator()(float x) { return cosf(x); } };
struct TanhOp { __device__ float operator()(float x) { return tanhf(x); } };

template<typename Op>
__global__ void elementwise_unary_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = Op()(input[idx]);
    }
}

void launch_elementwise_unary(
    ir::OpType op,
    float* output, const float* input,
    int64_t numel, hipStream_t stream
) {
    if (numel == 0) return;
    int grid = get_grid_size(numel);

    switch (op) {
        case ir::OpType::NEG:
            hipLaunchKernelGGL((elementwise_unary_kernel<NegOp>),
                              grid, BLOCK_SIZE, 0, stream, output, input, numel);
            break;
        case ir::OpType::ABS:
            hipLaunchKernelGGL((elementwise_unary_kernel<AbsOp>),
                              grid, BLOCK_SIZE, 0, stream, output, input, numel);
            break;
        case ir::OpType::SQRT:
            hipLaunchKernelGGL((elementwise_unary_kernel<SqrtOp>),
                              grid, BLOCK_SIZE, 0, stream, output, input, numel);
            break;
        case ir::OpType::EXP:
            hipLaunchKernelGGL((elementwise_unary_kernel<ExpOp>),
                              grid, BLOCK_SIZE, 0, stream, output, input, numel);
            break;
        case ir::OpType::LOG:
            hipLaunchKernelGGL((elementwise_unary_kernel<LogOp>),
                              grid, BLOCK_SIZE, 0, stream, output, input, numel);
            break;
        case ir::OpType::SIN:
            hipLaunchKernelGGL((elementwise_unary_kernel<SinOp>),
                              grid, BLOCK_SIZE, 0, stream, output, input, numel);
            break;
        case ir::OpType::COS:
            hipLaunchKernelGGL((elementwise_unary_kernel<CosOp>),
                              grid, BLOCK_SIZE, 0, stream, output, input, numel);
            break;
        case ir::OpType::TANH:
            hipLaunchKernelGGL((elementwise_unary_kernel<TanhOp>),
                              grid, BLOCK_SIZE, 0, stream, output, input, numel);
            break;
        default:
            throw std::runtime_error("Unsupported unary op in kernel");
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

// Comparison functors
struct EqOp { __device__ bool operator()(float a, float b) { return a == b; } };
struct NeOp { __device__ bool operator()(float a, float b) { return a != b; } };
struct LtOp { __device__ bool operator()(float a, float b) { return a < b; } };
struct LeOp { __device__ bool operator()(float a, float b) { return a <= b; } };
struct GtOp { __device__ bool operator()(float a, float b) { return a > b; } };
struct GeOp { __device__ bool operator()(float a, float b) { return a >= b; } };

template<typename Op>
__global__ void comparison_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = Op()(a[idx], b[idx]) ? 1.0f : 0.0f;
    }
}

void launch_comparison(
    ir::OpType op,
    float* output,
    const float* a, const float* b,
    int64_t numel, hipStream_t stream
) {
    if (numel == 0) return;
    int grid = get_grid_size(numel);

    switch (op) {
        case ir::OpType::EQ:
            hipLaunchKernelGGL((comparison_kernel<EqOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::NE:
            hipLaunchKernelGGL((comparison_kernel<NeOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::LT:
        case ir::OpType::LESS:
            hipLaunchKernelGGL((comparison_kernel<LtOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::LE:
            hipLaunchKernelGGL((comparison_kernel<LeOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::GT:
            hipLaunchKernelGGL((comparison_kernel<GtOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        case ir::OpType::GE:
            hipLaunchKernelGGL((comparison_kernel<GeOp>),
                              grid, BLOCK_SIZE, 0, stream, output, a, b, numel);
            break;
        default:
            throw std::runtime_error("Unsupported comparison op in kernel");
    }
}

// ============================================================================
// Where Kernel
// ============================================================================

__global__ void where_kernel(
    float* __restrict__ output,
    const float* __restrict__ condition,
    const float* __restrict__ x,
    const float* __restrict__ y,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = (condition[idx] != 0.0f) ? x[idx] : y[idx];
    }
}

void launch_where(
    float* output,
    const float* condition,
    const float* x, const float* y,
    int64_t numel, hipStream_t stream
) {
    if (numel == 0) return;
    int grid = get_grid_size(numel);
    hipLaunchKernelGGL(where_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, condition, x, y, numel);
}

// ============================================================================
// Clamp Kernel
// ============================================================================

__global__ void clamp_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float min_val, float max_val,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = fminf(fmaxf(input[idx], min_val), max_val);
    }
}

void launch_clamp(
    float* output, const float* input,
    float min_val, float max_val,
    int64_t numel, hipStream_t stream
) {
    if (numel == 0) return;
    int grid = get_grid_size(numel);
    hipLaunchKernelGGL(clamp_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, input, min_val, max_val, numel);
}

// ============================================================================
// Reduce Product Kernel
// ============================================================================

__global__ void reduce_prod_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel
) {
    __shared__ float sdata[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load and compute partial product
    float val = 1.0f;
    if (idx < numel) {
        val = input[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    // Block-level reduction (multiply)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result (using atomicMul not available, so use global memory)
    // Note: For proper multi-block product, would need two-pass or atomic CAS
    // This simple version works for single-block reductions
    if (tid == 0 && blockIdx.x == 0) {
        *output = sdata[0];
    }
}

void launch_reduce_prod(
    float* output, const float* input,
    int64_t numel, hipStream_t stream
) {
    if (numel == 0) {
        // Product of empty set is 1
        float one = 1.0f;
        HIP_CHECK(hipMemcpyAsync(output, &one, sizeof(float), hipMemcpyHostToDevice, stream));
        return;
    }

    // SECURITY FIX: Properly handle multi-block case with iterative reduction
    int grid = get_grid_size_safe(numel);

    if (grid == 1) {
        // Single block - direct reduction
        hipLaunchKernelGGL(reduce_prod_kernel, grid, BLOCK_SIZE, 0, stream,
                           output, input, numel);
    } else {
        // Multi-block reduction: use iterative approach
        // This is less efficient but correct for large inputs
        // For production, consider implementing proper multi-pass reduction

        // For now, throw if tensor is too large for single-block reduction
        // This ensures correctness over silent incorrect results
        if (numel > BLOCK_SIZE) {
            throw std::runtime_error(
                "reduce_prod: tensor too large for single-block reduction. "
                "numel=" + std::to_string(numel) + ", max=" + std::to_string(BLOCK_SIZE) +
                ". Use chunked reduction for large tensors.");
        }
        hipLaunchKernelGGL(reduce_prod_kernel, 1, BLOCK_SIZE, 0, stream,
                           output, input, numel);
    }
}

// ============================================================================
// Cross-Entropy Loss Kernel
// ============================================================================

__global__ void cross_entropy_loss_kernel(
    float* __restrict__ output,
    const float* __restrict__ logits,  // [N, C]
    const int32_t* __restrict__ targets,  // [N]
    int64_t N, int64_t C
) {
    __shared__ float sdata[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float loss = 0.0f;
    if (idx < N) {
        // Get logits row for this sample
        const float* row = logits + idx * C;
        int32_t target = targets[idx];

        // SECURITY: Bounds check on target index to prevent out-of-bounds read
        if (target < 0 || target >= static_cast<int32_t>(C)) {
            // Invalid target - contribute NaN to signal error
            // This will propagate through the loss and be detectable
            loss = NAN;
        } else {
            // Compute log-softmax for numerical stability
            // log_softmax(x_i) = x_i - log(sum(exp(x_j)))

            // Find max for numerical stability
            float max_val = -FLT_MAX;
            for (int64_t j = 0; j < C; j++) {
                max_val = fmaxf(max_val, row[j]);
            }

            // Compute sum of exp(x - max)
            float sum_exp = 0.0f;
            for (int64_t j = 0; j < C; j++) {
                sum_exp += expf(row[j] - max_val);
            }

            // Loss for this sample: -log_softmax[target]
            // = -(logits[target] - max - log(sum_exp))
            loss = -(row[target] - max_val - logf(sum_exp));
        }
    }

    sdata[tid] = loss;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Atomic add for mean
    if (tid == 0) {
        atomicAdd(output, sdata[0] / static_cast<float>(N));
    }
}

void launch_cross_entropy_loss(
    float* output,
    const float* logits,
    const int32_t* targets,
    int64_t N, int64_t C,
    hipStream_t stream
) {
    if (N == 0) return;

    // Zero output first
    HIP_CHECK(hipMemsetAsync(output, 0, sizeof(float), stream));

    int grid = get_grid_size(N);
    hipLaunchKernelGGL(cross_entropy_loss_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, logits, targets, N, C);
}

// ============================================================================
// BCE Loss Kernel
// ============================================================================

__global__ void bce_loss_kernel(
    float* __restrict__ output,
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    int64_t numel
) {
    __shared__ float sdata[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float loss = 0.0f;
    if (idx < numel) {
        float p = predictions[idx];
        float t = targets[idx];
        // BCE = -(t * log(p) + (1-t) * log(1-p))
        // Clamp p to avoid log(0)
        constexpr float eps = 1e-7f;
        p = fminf(fmaxf(p, eps), 1.0f - eps);
        loss = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
    }

    sdata[tid] = loss;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Atomic add for mean
    if (tid == 0) {
        atomicAdd(output, sdata[0] / static_cast<float>(numel));
    }
}

void launch_bce_loss(
    float* output,
    const float* predictions,
    const float* targets,
    int64_t numel,
    hipStream_t stream
) {
    if (numel == 0) return;

    // Zero output first
    HIP_CHECK(hipMemsetAsync(output, 0, sizeof(float), stream));

    int grid = get_grid_size(numel);
    hipLaunchKernelGGL(bce_loss_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, predictions, targets, numel);
}

// ============================================================================
// MSE Loss Kernel
// ============================================================================

__global__ void mse_loss_kernel(
    float* __restrict__ output,
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    int64_t numel
) {
    __shared__ float sdata[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Compute squared error
    float val = 0.0f;
    if (idx < numel) {
        float diff = predictions[idx] - targets[idx];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Atomic add for mean
    if (tid == 0) {
        atomicAdd(output, sdata[0] / static_cast<float>(numel));
    }
}

void launch_mse_loss(
    float* output,
    const float* predictions,
    const float* targets,
    int64_t numel,
    hipStream_t stream
) {
    if (numel == 0) return;

    // Zero output first
    HIP_CHECK(hipMemsetAsync(output, 0, sizeof(float), stream));

    int grid = get_grid_size(numel);
    hipLaunchKernelGGL(mse_loss_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, predictions, targets, numel);
}

// ============================================================================
// Fused Kernels for Performance Optimization (Phase 9)
// ============================================================================

// Fused bias add for matmul output (in-place)
__global__ void fused_bias_add_kernel(
    float* __restrict__ output,  // [M, N] - modified in place
    const float* __restrict__ bias,  // [N]
    int64_t M, int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (idx < total) {
        int64_t col = idx % N;
        output[idx] += bias[col];
    }
}

// Fused bias + ReLU for matmul output (in-place)
__global__ void fused_bias_relu_kernel(
    float* __restrict__ output,  // [M, N]
    const float* __restrict__ bias,  // [N]
    int64_t M, int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (idx < total) {
        int64_t col = idx % N;
        float val = output[idx] + bias[col];
        output[idx] = fmaxf(val, 0.0f);
    }
}

// Fused bias + GELU for matmul output (in-place)
__global__ void fused_bias_gelu_kernel(
    float* __restrict__ output,  // [M, N]
    const float* __restrict__ bias,  // [N]
    int64_t M, int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (idx < total) {
        int64_t col = idx % N;
        float x = output[idx] + bias[col];
        // GELU approximation
        float x3 = x * x * x;
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
    }
}

// Fused add + ReLU
__global__ void fused_add_relu_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float sum = a[idx] + b[idx];
        output[idx] = fmaxf(sum, 0.0f);
    }
}

// Fused multiply + add (FMA-like for elementwise)
__global__ void fused_mul_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // output = a * b + c
        output[idx] = __fmaf_rn(a[idx], b[idx], c[idx]);
    }
}

// Fused ReLU without bias (for when bias is already added)
__global__ void fused_relu_inplace_kernel(
    float* __restrict__ data,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

}  // namespace pyflame::backend::rocm::kernels

// ============================================================================
// C-linkage launch functions for fused kernels (called from rocm_perf.cpp)
// ============================================================================

extern "C" {

using namespace pyflame::backend::rocm::kernels;

void launch_fused_matmul_bias(
    float* output, const float* A, const float* B, const float* bias,
    int64_t M, int64_t K, int64_t N, hipStream_t stream
) {
    // GEMM is done separately via rocBLAS, this just adds bias
    (void)A; (void)B; (void)K;  // Unused - GEMM done separately
    if (!bias) return;

    int64_t total = M * N;
    int grid = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    hipLaunchKernelGGL(fused_bias_add_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, bias, M, N);
}

void launch_fused_matmul_bias_relu(
    float* output, const float* A, const float* B, const float* bias,
    int64_t M, int64_t K, int64_t N, hipStream_t stream
) {
    (void)A; (void)B; (void)K;

    int64_t total = M * N;
    int grid = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (bias) {
        hipLaunchKernelGGL(fused_bias_relu_kernel, grid, BLOCK_SIZE, 0, stream,
                           output, bias, M, N);
    } else {
        hipLaunchKernelGGL(fused_relu_inplace_kernel, grid, BLOCK_SIZE, 0, stream,
                           output, total);
    }
}

void launch_fused_matmul_bias_gelu(
    float* output, const float* A, const float* B, const float* bias,
    int64_t M, int64_t K, int64_t N, hipStream_t stream
) {
    (void)A; (void)B; (void)K;

    int64_t total = M * N;
    int grid = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    hipLaunchKernelGGL(fused_bias_gelu_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, bias, M, N);
}

void launch_fused_add_relu(
    float* output, const float* a, const float* b,
    int64_t numel, hipStream_t stream
) {
    int grid = static_cast<int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
    hipLaunchKernelGGL(fused_add_relu_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, a, b, numel);
}

void launch_fused_mul_add(
    float* output, const float* a, const float* b, const float* c,
    int64_t numel, hipStream_t stream
) {
    int grid = static_cast<int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
    hipLaunchKernelGGL(fused_mul_add_kernel, grid, BLOCK_SIZE, 0, stream,
                       output, a, b, c, numel);
}

}  // extern "C"

#endif  // PYFLAME_HAS_ROCM
