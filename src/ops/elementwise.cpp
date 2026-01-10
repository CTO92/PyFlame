// Elementwise operations CPU reference implementation
// This file contains optimized CPU implementations for development and testing
// The actual WSE implementations use CSL code generation

#include "pyflame/core/tensor.hpp"
#include "pyflame/core/tensor_impl.hpp"
#include "pyflame/core/safe_math.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <functional>
#include <cassert>

namespace pyflame {
namespace ops {

// ============================================================================
// Debug bounds checking (enabled in debug builds)
// ============================================================================

#ifdef NDEBUG
#define PYFLAME_BOUNDS_CHECK(ptr, n, idx) ((void)0)
#define PYFLAME_VALIDATE_SIZE(n) ((void)0)
#else
#define PYFLAME_BOUNDS_CHECK(ptr, n, idx) \
    do { \
        if ((idx) < 0 || (idx) >= (n)) { \
            throw std::out_of_range("Array index out of bounds: " + \
                std::to_string(idx) + " >= " + std::to_string(n)); \
        } \
    } while(0)

#define PYFLAME_VALIDATE_SIZE(n) \
    do { \
        if ((n) < 0) { \
            throw std::invalid_argument("Invalid size: " + std::to_string(n)); \
        } \
        if ((n) > MAX_TENSOR_NUMEL) { \
            throw std::overflow_error("Size too large: " + std::to_string(n)); \
        } \
    } while(0)
#endif

// ============================================================================
// Configuration
// ============================================================================

/// Enable strict mode for numerical operations (check for NaN/Inf)
#ifndef PYFLAME_STRICT_NUMERICS
#define PYFLAME_STRICT_NUMERICS 0
#endif

/// Small epsilon for numerical stability
constexpr float EPSILON = 1e-7f;

/// Check if a float value is valid (not NaN or Inf)
inline bool is_valid_float(float x) {
    return std::isfinite(x);
}

/// Validate result in strict mode
inline void validate_result([[maybe_unused]] float result, [[maybe_unused]] const char* op) {
#if PYFLAME_STRICT_NUMERICS
    if (!is_valid_float(result)) {
        throw std::runtime_error(std::string("Invalid result in ") + op +
            ": got " + (std::isnan(result) ? "NaN" : "Inf"));
    }
#endif
}

// ============================================================================
// Basic Arithmetic Operations
// ============================================================================

void add_cpu(const float* a, const float* b, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
        validate_result(out[i], "add");
    }
}

void sub_cpu(const float* a, const float* b, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
        validate_result(out[i], "sub");
    }
}

void mul_cpu(const float* a, const float* b, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
        validate_result(out[i], "mul");
    }
}

void div_cpu(const float* a, const float* b, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        if (b[i] == 0.0f) {
            // Handle division by zero gracefully
            // Following IEEE 754: x/0 = +/-inf for x != 0, NaN for 0/0
            if (a[i] == 0.0f) {
                out[i] = std::numeric_limits<float>::quiet_NaN();
            } else if (a[i] > 0.0f) {
                out[i] = std::numeric_limits<float>::infinity();
            } else {
                out[i] = -std::numeric_limits<float>::infinity();
            }
#if PYFLAME_STRICT_NUMERICS
            throw std::runtime_error("Division by zero at index " + std::to_string(i));
#endif
        } else {
            out[i] = a[i] / b[i];
        }
        validate_result(out[i], "div");
    }
}

/// Safe division with epsilon to avoid division by zero
void safe_div_cpu(const float* a, const float* b, float* out, int64_t n, float eps = EPSILON) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        float divisor = b[i];
        // Add epsilon to avoid division by zero, preserving sign
        if (std::abs(divisor) < eps) {
            divisor = (divisor >= 0.0f) ? eps : -eps;
        }
        out[i] = a[i] / divisor;
        validate_result(out[i], "safe_div");
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

void neg_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = -x[i];
    }
}

void abs_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::abs(x[i]);
    }
}

void sqrt_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        if (x[i] < 0.0f) {
            // sqrt of negative number
#if PYFLAME_STRICT_NUMERICS
            throw std::domain_error("sqrt of negative number at index " + std::to_string(i) +
                ": " + std::to_string(x[i]));
#else
            out[i] = std::numeric_limits<float>::quiet_NaN();
#endif
        } else {
            out[i] = std::sqrt(x[i]);
        }
        validate_result(out[i], "sqrt");
    }
}

void exp_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    // exp(88.7) > FLT_MAX, so we clamp input to avoid overflow
    constexpr float EXP_MAX_INPUT = 88.0f;

    for (int64_t i = 0; i < n; ++i) {
        float val = x[i];
        if (val > EXP_MAX_INPUT) {
#if PYFLAME_STRICT_NUMERICS
            throw std::overflow_error("exp overflow at index " + std::to_string(i) +
                ": input " + std::to_string(val) + " > " + std::to_string(EXP_MAX_INPUT));
#else
            out[i] = std::numeric_limits<float>::infinity();
#endif
        } else {
            out[i] = std::exp(val);
        }
        validate_result(out[i], "exp");
    }
}

void log_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        if (x[i] <= 0.0f) {
            // log of non-positive number
#if PYFLAME_STRICT_NUMERICS
            throw std::domain_error("log of non-positive number at index " + std::to_string(i) +
                ": " + std::to_string(x[i]));
#else
            if (x[i] == 0.0f) {
                out[i] = -std::numeric_limits<float>::infinity();
            } else {
                out[i] = std::numeric_limits<float>::quiet_NaN();
            }
#endif
        } else {
            out[i] = std::log(x[i]);
        }
        validate_result(out[i], "log");
    }
}

/// Safe log with clamping to avoid log(0)
void safe_log_cpu(const float* x, float* out, int64_t n, float eps = EPSILON) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        float val = std::max(x[i], eps);
        out[i] = std::log(val);
        validate_result(out[i], "safe_log");
    }
}

void sin_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::sin(x[i]);
        validate_result(out[i], "sin");
    }
}

void cos_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::cos(x[i]);
        validate_result(out[i], "cos");
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

void relu_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
}

void sigmoid_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        // Numerically stable sigmoid
        float val = x[i];
        if (val >= 0) {
            float z = std::exp(-val);
            out[i] = 1.0f / (1.0f + z);
        } else {
            float z = std::exp(val);
            out[i] = z / (1.0f + z);
        }
        validate_result(out[i], "sigmoid");
    }
}

void tanh_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::tanh(x[i]);
        validate_result(out[i], "tanh");
    }
}

void gelu_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;

    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        float inner = sqrt_2_over_pi * (xi + coef * xi * xi * xi);
        out[i] = 0.5f * xi * (1.0f + std::tanh(inner));
        validate_result(out[i], "gelu");
    }
}

void silu_cpu(const float* x, float* out, int64_t n) {
    PYFLAME_VALIDATE_SIZE(n);
    // SiLU (Swish): x * sigmoid(x)
    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        // Use numerically stable sigmoid
        float sig;
        if (xi >= 0) {
            float z = std::exp(-xi);
            sig = 1.0f / (1.0f + z);
        } else {
            float z = std::exp(xi);
            sig = z / (1.0f + z);
        }
        out[i] = xi * sig;
        validate_result(out[i], "silu");
    }
}

void softmax_cpu(const float* x, float* out, int64_t batch, int64_t dim) {
    // Validate sizes to prevent overflow
    PYFLAME_VALIDATE_SIZE(batch);
    PYFLAME_VALIDATE_SIZE(dim);

    // Check that batch * dim doesn't overflow
    int64_t total;
    if (safe_mul_overflow(batch, dim, &total)) {
        throw std::overflow_error("Softmax size overflow: batch=" + std::to_string(batch) +
            " * dim=" + std::to_string(dim) + " overflows int64_t");
    }

    if (dim <= 0) {
        throw std::invalid_argument("Softmax dimension must be positive, got " + std::to_string(dim));
    }

    // Softmax over last dimension
    for (int64_t b = 0; b < batch; ++b) {
        const float* row = x + b * dim;
        float* out_row = out + b * dim;

        // Find max for numerical stability
        float max_val = row[0];
        for (int64_t i = 1; i < dim; ++i) {
            max_val = std::max(max_val, row[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            out_row[i] = std::exp(row[i] - max_val);
            sum += out_row[i];
        }

        // Normalize with epsilon to avoid division by zero
        if (sum < EPSILON) {
            sum = EPSILON;
        }
        for (int64_t i = 0; i < dim; ++i) {
            out_row[i] /= sum;
            validate_result(out_row[i], "softmax");
        }
    }
}

void log_softmax_cpu(const float* x, float* out, int64_t batch, int64_t dim) {
    // Validate sizes
    PYFLAME_VALIDATE_SIZE(batch);
    PYFLAME_VALIDATE_SIZE(dim);

    int64_t total;
    if (safe_mul_overflow(batch, dim, &total)) {
        throw std::overflow_error("Log-softmax size overflow");
    }

    if (dim <= 0) {
        throw std::invalid_argument("Log-softmax dimension must be positive");
    }

    // Log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    for (int64_t b = 0; b < batch; ++b) {
        const float* row = x + b * dim;
        float* out_row = out + b * dim;

        // Find max for numerical stability
        float max_val = row[0];
        for (int64_t i = 1; i < dim; ++i) {
            max_val = std::max(max_val, row[i]);
        }

        // Compute log-sum-exp
        float sum_exp = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            sum_exp += std::exp(row[i] - max_val);
        }
        float log_sum_exp = std::log(std::max(sum_exp, EPSILON)) + max_val;

        // Compute log-softmax
        for (int64_t i = 0; i < dim; ++i) {
            out_row[i] = row[i] - log_sum_exp;
            validate_result(out_row[i], "log_softmax");
        }
    }
}

// ============================================================================
// Broadcasting Helpers
// ============================================================================

void broadcast_binary_op(
    const float* a, const std::vector<int64_t>& a_shape,
    const float* b, const std::vector<int64_t>& b_shape,
    float* out, const std::vector<int64_t>& out_shape,
    std::function<float(float, float)> op
) {
    // Compute total elements with overflow checking
    int64_t n = safe_numel(out_shape);
    PYFLAME_VALIDATE_SIZE(n);

    // If shapes match, simple elementwise
    if (a_shape == b_shape) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = op(a[i], b[i]);
            validate_result(out[i], "broadcast_op");
        }
        return;
    }

    // Handle broadcasting (simplified - assumes contiguous memory)
    // Full implementation would compute strides properly
    int64_t a_numel = safe_numel(a_shape);
    int64_t b_numel = safe_numel(b_shape);

    for (int64_t i = 0; i < n; ++i) {
        // Compute indices in each tensor with modulo for broadcasting
        int64_t a_idx = (a_numel > 0) ? (i % a_numel) : 0;
        int64_t b_idx = (b_numel > 0) ? (i % b_numel) : 0;
        out[i] = op(a[a_idx], b[b_idx]);
        validate_result(out[i], "broadcast_op");
    }
}

// ============================================================================
// Clamp Operation
// ============================================================================

void clamp_cpu(const float* x, float* out, int64_t n, float min_val, float max_val) {
    PYFLAME_VALIDATE_SIZE(n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::clamp(x[i], min_val, max_val);
    }
}

}  // namespace ops
}  // namespace pyflame
