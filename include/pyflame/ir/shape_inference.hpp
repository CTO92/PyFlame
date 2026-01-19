#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <optional>

#include "pyflame/ir/node.hpp"
#include "pyflame/ir/op_type.hpp"

namespace pyflame::ir {

/// Shape inference utilities

/// Compute broadcasted shape from two input shapes
inline std::vector<int64_t> broadcast_shapes(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
) {
    size_t max_ndim = std::max(a.size(), b.size());
    std::vector<int64_t> result(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        int64_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (dim_a == dim_b) {
            result[max_ndim - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            result[max_ndim - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            result[max_ndim - 1 - i] = dim_a;
        } else {
            throw std::runtime_error(
                "Shapes are not broadcastable: dimension " +
                std::to_string(dim_a) + " vs " + std::to_string(dim_b)
            );
        }
    }

    return result;
}

/// Infer output spec for binary elementwise operations
inline TensorSpec infer_binary_op_spec(
    const TensorSpec& a,
    const TensorSpec& b,
    OpType op
) {
    // Check dtype compatibility
    if (a.dtype != b.dtype) {
        throw std::runtime_error("Binary operation requires matching dtypes");
    }

    // Compute broadcasted shape
    auto shape = broadcast_shapes(a.shape, b.shape);

    // Use layout from first operand (may need transformation)
    return TensorSpec(shape, a.dtype, a.layout);
}

/// Infer output spec for unary operations
inline TensorSpec infer_unary_op_spec(
    const TensorSpec& input,
    OpType op
) {
    // Most unary ops preserve shape, dtype, and layout
    return input;
}

/// Infer output spec for reduction operations
inline TensorSpec infer_reduction_spec(
    const TensorSpec& input,
    std::optional<int> dim,
    bool keepdim
) {
    if (!dim.has_value()) {
        // Full reduction to scalar
        if (keepdim) {
            std::vector<int64_t> shape(input.shape.size(), 1);
            return TensorSpec(shape, input.dtype, MeshLayout::SinglePE());
        } else {
            return TensorSpec({}, input.dtype, MeshLayout::SinglePE());
        }
    }

    int d = dim.value();
    if (d < 0) d += static_cast<int>(input.shape.size());

    if (d < 0 || d >= static_cast<int>(input.shape.size())) {
        throw std::runtime_error("Reduction dimension out of range");
    }

    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < input.shape.size(); ++i) {
        if (static_cast<int>(i) == d) {
            if (keepdim) {
                new_shape.push_back(1);
            }
        } else {
            new_shape.push_back(input.shape[i]);
        }
    }

    return TensorSpec(new_shape, input.dtype, input.layout);
}

/// Infer output spec for matmul
inline TensorSpec infer_matmul_spec(
    const TensorSpec& a,
    const TensorSpec& b
) {
    if (a.shape.empty() || b.shape.empty()) {
        throw std::runtime_error("matmul requires non-scalar inputs");
    }

    // Check dtype compatibility
    if (a.dtype != b.dtype) {
        throw std::runtime_error("matmul requires matching dtypes");
    }

    // 1D @ 1D -> scalar
    if (a.shape.size() == 1 && b.shape.size() == 1) {
        if (a.shape[0] != b.shape[0]) {
            throw std::runtime_error("matmul: incompatible shapes for dot product");
        }
        return TensorSpec({}, a.dtype, MeshLayout::SinglePE());
    }

    // 2D @ 2D -> 2D
    if (a.shape.size() == 2 && b.shape.size() == 2) {
        if (a.shape[1] != b.shape[0]) {
            throw std::runtime_error(
                "matmul: shapes not aligned: " +
                std::to_string(a.shape[1]) + " vs " + std::to_string(b.shape[0])
            );
        }
        return TensorSpec({a.shape[0], b.shape[1]}, a.dtype, a.layout);
    }

    // 1D @ 2D -> 1D
    if (a.shape.size() == 1 && b.shape.size() == 2) {
        if (a.shape[0] != b.shape[0]) {
            throw std::runtime_error("matmul: shapes not aligned");
        }
        return TensorSpec({b.shape[1]}, a.dtype, a.layout);
    }

    // 2D @ 1D -> 1D
    if (a.shape.size() == 2 && b.shape.size() == 1) {
        if (a.shape[1] != b.shape[0]) {
            throw std::runtime_error("matmul: shapes not aligned");
        }
        return TensorSpec({a.shape[0]}, a.dtype, a.layout);
    }

    // Batched matmul
    // Get the batch dimensions
    std::vector<int64_t> a_batch(a.shape.begin(), a.shape.end() - 2);
    std::vector<int64_t> b_batch(b.shape.begin(), b.shape.end() - 2);

    // Broadcast batch dimensions
    auto batch_shape = broadcast_shapes(a_batch, b_batch);

    // Check matrix dimensions
    int64_t a_m = a.shape[a.shape.size() - 2];
    int64_t a_k = a.shape[a.shape.size() - 1];
    int64_t b_k = b.shape[b.shape.size() - 2];
    int64_t b_n = b.shape[b.shape.size() - 1];

    if (a_k != b_k) {
        throw std::runtime_error("matmul: inner dimensions don't match");
    }

    std::vector<int64_t> result_shape = batch_shape;
    result_shape.push_back(a_m);
    result_shape.push_back(b_n);

    return TensorSpec(result_shape, a.dtype, a.layout);
}

/// Infer output spec for transpose
inline TensorSpec infer_transpose_spec(
    const TensorSpec& input,
    int dim0,
    int dim1
) {
    int ndim = static_cast<int>(input.shape.size());

    // Handle negative dimensions
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::runtime_error("transpose: dimension out of range");
    }

    std::vector<int64_t> new_shape = input.shape;
    std::swap(new_shape[dim0], new_shape[dim1]);

    return TensorSpec(new_shape, input.dtype, input.layout);
}

/// Infer output spec for reshape
inline TensorSpec infer_reshape_spec(
    const TensorSpec& input,
    const std::vector<int64_t>& new_shape
) {
    // Compute total elements
    int64_t input_numel = input.numel();

    // Handle -1 dimension
    int64_t output_numel = 1;
    int neg_one_idx = -1;

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (neg_one_idx >= 0) {
                throw std::runtime_error("reshape: only one -1 allowed");
            }
            neg_one_idx = static_cast<int>(i);
        } else if (new_shape[i] < 0) {
            throw std::runtime_error("reshape: invalid dimension");
        } else {
            output_numel *= new_shape[i];
        }
    }

    std::vector<int64_t> result_shape = new_shape;

    if (neg_one_idx >= 0) {
        if (output_numel == 0 || input_numel % output_numel != 0) {
            throw std::runtime_error("reshape: cannot infer -1 dimension");
        }
        result_shape[neg_one_idx] = input_numel / output_numel;
    } else {
        if (output_numel != input_numel) {
            throw std::runtime_error("reshape: total elements must match");
        }
    }

    return TensorSpec(result_shape, input.dtype, input.layout);
}

/// Infer output spec for concatenation
inline TensorSpec infer_concat_spec(
    const std::vector<TensorSpec>& inputs,
    int dim
) {
    if (inputs.empty()) {
        throw std::runtime_error("concat: requires at least one input");
    }

    const TensorSpec& first = inputs[0];
    int ndim = first.ndim();

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("concat: dimension out of range");
    }

    std::vector<int64_t> result_shape = first.shape;
    int64_t concat_dim_size = first.shape[dim];

    for (size_t i = 1; i < inputs.size(); ++i) {
        const TensorSpec& input = inputs[i];

        if (input.dtype != first.dtype) {
            throw std::runtime_error("concat: all inputs must have same dtype");
        }

        if (input.ndim() != ndim) {
            throw std::runtime_error("concat: all inputs must have same number of dimensions");
        }

        for (int d = 0; d < ndim; ++d) {
            if (d == dim) {
                concat_dim_size += input.shape[d];
            } else if (input.shape[d] != first.shape[d]) {
                throw std::runtime_error("concat: shapes don't match on non-concat dimension");
            }
        }
    }

    result_shape[dim] = concat_dim_size;
    return TensorSpec(result_shape, first.dtype, first.layout);
}

/// Infer output spec for softmax
inline TensorSpec infer_softmax_spec(
    const TensorSpec& input,
    int dim
) {
    // Softmax preserves shape
    return input;
}

/// Infer output spec for slice operation
inline TensorSpec infer_slice_spec(
    const TensorSpec& input,
    int dim,
    int64_t start,
    int64_t end
) {
    int ndim = static_cast<int>(input.shape.size());

    // Handle negative dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("slice: dimension out of range");
    }

    int64_t dim_size = input.shape[dim];

    // Handle negative indices
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;

    // Clamp to valid range
    start = std::max(int64_t(0), std::min(start, dim_size));
    end = std::max(int64_t(0), std::min(end, dim_size));

    if (end <= start) {
        throw std::runtime_error("slice: end must be greater than start");
    }

    std::vector<int64_t> new_shape = input.shape;
    new_shape[dim] = end - start;

    return TensorSpec(new_shape, input.dtype, input.layout);
}

/// Infer output spec for dtype conversion (cast)
inline TensorSpec infer_cast_spec(
    const TensorSpec& input,
    DType new_dtype
) {
    return TensorSpec(input.shape, new_dtype, input.layout);
}

}  // namespace pyflame::ir
