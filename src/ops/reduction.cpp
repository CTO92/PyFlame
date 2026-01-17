// Reduction operations CPU reference implementation

#include "pyflame/core/tensor.hpp"
#include <algorithm>
#include <numeric>
#include <limits>

namespace pyflame {
namespace ops {

// Full reduction (to scalar)
float sum_all_cpu(const float* x, int64_t n) {
    float result = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        result += x[i];
    }
    return result;
}

float mean_all_cpu(const float* x, int64_t n) {
    return sum_all_cpu(x, n) / static_cast<float>(n);
}

float max_all_cpu(const float* x, int64_t n) {
    float result = x[0];
    for (int64_t i = 1; i < n; ++i) {
        result = std::max(result, x[i]);
    }
    return result;
}

float min_all_cpu(const float* x, int64_t n) {
    float result = x[0];
    for (int64_t i = 1; i < n; ++i) {
        result = std::min(result, x[i]);
    }
    return result;
}

float prod_all_cpu(const float* x, int64_t n) {
    float result = 1.0f;
    for (int64_t i = 0; i < n; ++i) {
        result *= x[i];
    }
    return result;
}

// Reduction along a dimension (for 2D tensors)
void sum_dim_cpu(
    const float* x,
    float* out,
    int64_t dim0,
    int64_t dim1,
    int reduce_dim,  // 0 or 1
    bool keepdim
) {
    if (reduce_dim == 0) {
        // Sum along rows -> output shape (1, dim1) or (dim1,)
        for (int64_t j = 0; j < dim1; ++j) {
            float sum = 0.0f;
            for (int64_t i = 0; i < dim0; ++i) {
                sum += x[i * dim1 + j];
            }
            out[j] = sum;
        }
    } else {
        // Sum along cols -> output shape (dim0, 1) or (dim0,)
        for (int64_t i = 0; i < dim0; ++i) {
            float sum = 0.0f;
            for (int64_t j = 0; j < dim1; ++j) {
                sum += x[i * dim1 + j];
            }
            out[i] = sum;
        }
    }
}

void mean_dim_cpu(
    const float* x,
    float* out,
    int64_t dim0,
    int64_t dim1,
    int reduce_dim,
    bool keepdim
) {
    sum_dim_cpu(x, out, dim0, dim1, reduce_dim, keepdim);

    int64_t reduce_size = (reduce_dim == 0) ? dim0 : dim1;
    int64_t out_size = (reduce_dim == 0) ? dim1 : dim0;

    for (int64_t i = 0; i < out_size; ++i) {
        out[i] /= static_cast<float>(reduce_size);
    }
}

void max_dim_cpu(
    const float* x,
    float* out,
    int64_t dim0,
    int64_t dim1,
    int reduce_dim,
    bool keepdim
) {
    if (reduce_dim == 0) {
        for (int64_t j = 0; j < dim1; ++j) {
            float max_val = x[j];
            for (int64_t i = 1; i < dim0; ++i) {
                max_val = std::max(max_val, x[i * dim1 + j]);
            }
            out[j] = max_val;
        }
    } else {
        for (int64_t i = 0; i < dim0; ++i) {
            float max_val = x[i * dim1];
            for (int64_t j = 1; j < dim1; ++j) {
                max_val = std::max(max_val, x[i * dim1 + j]);
            }
            out[i] = max_val;
        }
    }
}

void min_dim_cpu(
    const float* x,
    float* out,
    int64_t dim0,
    int64_t dim1,
    int reduce_dim,
    bool keepdim
) {
    if (reduce_dim == 0) {
        for (int64_t j = 0; j < dim1; ++j) {
            float min_val = x[j];
            for (int64_t i = 1; i < dim0; ++i) {
                min_val = std::min(min_val, x[i * dim1 + j]);
            }
            out[j] = min_val;
        }
    } else {
        for (int64_t i = 0; i < dim0; ++i) {
            float min_val = x[i * dim1];
            for (int64_t j = 1; j < dim1; ++j) {
                min_val = std::min(min_val, x[i * dim1 + j]);
            }
            out[i] = min_val;
        }
    }
}

// General N-dimensional reduction
void reduce_nd_cpu(
    const float* x,
    float* out,
    const std::vector<int64_t>& shape,
    int reduce_dim,
    bool keepdim,
    float (*reduce_fn)(const float*, int64_t)
) {
    // Compute strides
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }

    // Compute output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int>(i) == reduce_dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }

    // Compute output size
    int64_t out_numel = 1;
    for (auto d : out_shape) out_numel *= d;

    // For each output position
    for (int64_t out_idx = 0; out_idx < out_numel; ++out_idx) {
        // Compute the multi-index for output
        std::vector<int64_t> out_multi(out_shape.size());
        int64_t tmp = out_idx;
        for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
            out_multi[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }

        // Map to input multi-index (insert reduce_dim = 0)
        std::vector<int64_t> in_multi_base(shape.size());
        int out_i = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int>(i) == reduce_dim) {
                in_multi_base[i] = 0;
            } else {
                in_multi_base[i] = out_multi[out_i++];
            }
        }

        // Collect values along the reduction dimension
        std::vector<float> values;
        values.reserve(shape[reduce_dim]);
        for (int64_t r = 0; r < shape[reduce_dim]; ++r) {
            in_multi_base[reduce_dim] = r;

            // Compute linear index
            int64_t in_idx = 0;
            for (size_t i = 0; i < shape.size(); ++i) {
                in_idx += in_multi_base[i] * strides[i];
            }

            values.push_back(x[in_idx]);
        }

        // Apply the reduction function
        out[out_idx] = reduce_fn(values.data(), static_cast<int64_t>(values.size()));
    }
}

}  // namespace ops
}  // namespace pyflame
