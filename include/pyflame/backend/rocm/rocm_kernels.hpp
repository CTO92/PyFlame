#pragma once

#ifdef PYFLAME_HAS_ROCM

#include <hip/hip_runtime.h>
#include "pyflame/core/dtype.hpp"
#include "pyflame/ir/op_type.hpp"
#include <cstdint>

namespace pyflame::backend::rocm::kernels {

// ============================================================================
// Activation Kernels
// ============================================================================

/// Launch GELU activation forward kernel
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// @param output Output tensor (device memory)
/// @param input Input tensor (device memory)
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_gelu_forward(
    float* output, const float* input,
    int64_t numel, hipStream_t stream);

/// Launch SiLU (Swish) activation forward kernel
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// @param output Output tensor (device memory)
/// @param input Input tensor (device memory)
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_silu_forward(
    float* output, const float* input,
    int64_t numel, hipStream_t stream);

// ============================================================================
// Elementwise Binary Kernels
// ============================================================================

/// Launch elementwise binary operation kernel
/// Supports: ADD, SUB, MUL, DIV, POW, MAX_BINARY, MIN_BINARY, MAXIMUM, MINIMUM
/// @param op Operation type
/// @param output Output tensor (device memory)
/// @param a First input tensor (device memory)
/// @param b Second input tensor (device memory)
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_elementwise_binary(
    ir::OpType op,
    float* output,
    const float* a, const float* b,
    int64_t numel, hipStream_t stream);

// ============================================================================
// Elementwise Unary Kernels
// ============================================================================

/// Launch elementwise unary operation kernel
/// Supports: NEG, ABS, SQRT, EXP, LOG, SIN, COS, TANH
/// @param op Operation type
/// @param output Output tensor (device memory)
/// @param input Input tensor (device memory)
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_elementwise_unary(
    ir::OpType op,
    float* output, const float* input,
    int64_t numel, hipStream_t stream);

// ============================================================================
// Comparison Kernels
// ============================================================================

/// Launch comparison operation kernel
/// Returns 1.0f for true, 0.0f for false
/// Supports: EQ, NE, LT, LE, GT, GE, LESS
/// @param op Comparison operation type
/// @param output Output tensor (device memory), stores 0.0 or 1.0
/// @param a First input tensor (device memory)
/// @param b Second input tensor (device memory)
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_comparison(
    ir::OpType op,
    float* output,
    const float* a, const float* b,
    int64_t numel, hipStream_t stream);

// ============================================================================
// Conditional Kernels
// ============================================================================

/// Launch where (conditional select) kernel
/// output[i] = condition[i] != 0 ? x[i] : y[i]
/// @param output Output tensor (device memory)
/// @param condition Condition tensor (device memory), 0.0 for false, non-zero for true
/// @param x Tensor for true case (device memory)
/// @param y Tensor for false case (device memory)
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_where(
    float* output,
    const float* condition,
    const float* x, const float* y,
    int64_t numel, hipStream_t stream);

/// Launch clamp kernel
/// output[i] = min(max(input[i], min_val), max_val)
/// @param output Output tensor (device memory)
/// @param input Input tensor (device memory)
/// @param min_val Minimum value
/// @param max_val Maximum value
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_clamp(
    float* output, const float* input,
    float min_val, float max_val,
    int64_t numel, hipStream_t stream);

// ============================================================================
// Reduction Kernels
// ============================================================================

/// Launch product reduction kernel
/// Computes the product of all elements
/// @param output Single output value (device memory)
/// @param input Input tensor (device memory)
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_reduce_prod(
    float* output, const float* input,
    int64_t numel, hipStream_t stream);

// ============================================================================
// Loss Function Kernels
// ============================================================================

/// Launch cross-entropy loss kernel
/// Computes: -sum(log(softmax(logits)[targets])) / N
/// @param output Single output loss value (device memory)
/// @param logits Logit tensor [N, C] (device memory)
/// @param targets Target indices [N] (device memory)
/// @param N Batch size
/// @param C Number of classes
/// @param stream HIP stream for async execution
void launch_cross_entropy_loss(
    float* output,
    const float* logits,
    const int32_t* targets,
    int64_t N, int64_t C,
    hipStream_t stream);

/// Launch binary cross-entropy loss kernel
/// Computes: -mean(targets * log(preds) + (1 - targets) * log(1 - preds))
/// @param output Single output loss value (device memory)
/// @param predictions Prediction tensor (device memory), values in [0, 1]
/// @param targets Target tensor (device memory), values in {0, 1}
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_bce_loss(
    float* output,
    const float* predictions,
    const float* targets,
    int64_t numel,
    hipStream_t stream);

/// Launch mean squared error loss kernel
/// Computes: mean((predictions - targets)^2)
/// @param output Single output loss value (device memory)
/// @param predictions Prediction tensor (device memory)
/// @param targets Target tensor (device memory)
/// @param numel Number of elements
/// @param stream HIP stream for async execution
void launch_mse_loss(
    float* output,
    const float* predictions,
    const float* targets,
    int64_t numel,
    hipStream_t stream);

}  // namespace pyflame::backend::rocm::kernels

#endif  // PYFLAME_HAS_ROCM
