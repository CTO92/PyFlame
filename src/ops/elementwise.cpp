// Elementwise operations CPU reference implementation
// This file contains optimized CPU implementations for development and testing
// The actual WSE implementations use CSL code generation

#include "pyflame/core/tensor.hpp"
#include "pyflame/core/tensor_impl.hpp"
#include <cmath>
#include <algorithm>

namespace pyflame {
namespace ops {

// These functions are called by the executor for CPU reference execution

void add_cpu(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sub_cpu(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void mul_cpu(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void div_cpu(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}

void neg_cpu(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = -x[i];
    }
}

void abs_cpu(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::abs(x[i]);
    }
}

void sqrt_cpu(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::sqrt(x[i]);
    }
}

void exp_cpu(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::exp(x[i]);
    }
}

void log_cpu(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::log(x[i]);
    }
}

void relu_cpu(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
}

void sigmoid_cpu(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

void tanh_cpu(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::tanh(x[i]);
    }
}

void gelu_cpu(const float* x, float* out, int64_t n) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;

    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        float inner = sqrt_2_over_pi * (xi + coef * xi * xi * xi);
        out[i] = 0.5f * xi * (1.0f + std::tanh(inner));
    }
}

void silu_cpu(const float* x, float* out, int64_t n) {
    // SiLU (Swish): x * sigmoid(x)
    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        out[i] = xi / (1.0f + std::exp(-xi));
    }
}

void softmax_cpu(const float* x, float* out, int64_t batch, int64_t dim) {
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

        // Normalize
        for (int64_t i = 0; i < dim; ++i) {
            out_row[i] /= sum;
        }
    }
}

// Broadcasting helpers
void broadcast_binary_op(
    const float* a, const std::vector<int64_t>& a_shape,
    const float* b, const std::vector<int64_t>& b_shape,
    float* out, const std::vector<int64_t>& out_shape,
    std::function<float(float, float)> op
) {
    // Simplified broadcasting for common cases
    int64_t n = 1;
    for (auto d : out_shape) n *= d;

    // If shapes match, simple elementwise
    if (a_shape == b_shape) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = op(a[i], b[i]);
        }
        return;
    }

    // Handle broadcasting (simplified - assumes contiguous memory)
    // Full implementation would compute strides properly
    for (int64_t i = 0; i < n; ++i) {
        // Compute indices in each tensor
        // This is simplified - real impl needs proper stride calculation
        out[i] = op(a[i % (a_shape.empty() ? 1 : a_shape[0])],
                    b[i % (b_shape.empty() ? 1 : b_shape[0])]);
    }
}

}  // namespace ops
}  // namespace pyflame
