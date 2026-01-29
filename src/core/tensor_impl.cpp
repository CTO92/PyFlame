#include "pyflame/core/tensor_impl.hpp"
#include "pyflame/ir/shape_inference.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>

namespace pyflame {

void* TensorImpl::materialize() {
    if (data_) {
        return data_.get();
    }

    // For now, we execute operations eagerly on the CPU
    // In the full implementation, this would compile to CSL and run on WSE

    if (node_->is_constant() && node_->has_constant_data()) {
        // Copy constant data
        size_t bytes = node_->output_spec().size_bytes();
        data_ = std::shared_ptr<uint8_t>(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );
        std::memcpy(data_.get(), node_->constant_data().data(), bytes);
        node_->mark_evaluated();
        return data_.get();
    }

    // Execute the operation on CPU (reference implementation)
    if (node_->is_operation()) {
        execute_cpu();
    }

    return data_.get();
}

// CPU reference implementation for operations
void TensorImpl::execute_cpu() {
    const auto& inputs = node_->inputs();
    auto op = node_->op_type();
    int64_t numel = node_->numel();
    size_t bytes = node_->output_spec().size_bytes();

    // Allocate output
    data_ = std::shared_ptr<uint8_t>(
        static_cast<uint8_t*>(Allocator::allocate(bytes)),
        [](uint8_t* p) { Allocator::deallocate(p); }
    );

    // Get input data (forces their materialization)
    std::vector<const float*> input_data;
    for (const auto& input : inputs) {
        // Find or create TensorImpl for input
        // For now, assume inputs are already materialized or are constants
        if (input->has_constant_data()) {
            input_data.push_back(input->constant_data_as<float>());
        } else {
            // This is a simplification - in real impl, we'd track TensorImpl for each node
            input_data.push_back(nullptr);
        }
    }

    float* output = reinterpret_cast<float*>(data_.get());

    switch (op) {
        // Binary elementwise ops
        case ir::OpType::ADD:
            if (input_data[0] && input_data[1]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = input_data[0][i] + input_data[1][i];
                }
            }
            break;

        case ir::OpType::SUB:
            if (input_data[0] && input_data[1]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = input_data[0][i] - input_data[1][i];
                }
            }
            break;

        case ir::OpType::MUL:
            if (input_data[0] && input_data[1]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = input_data[0][i] * input_data[1][i];
                }
            }
            break;

        case ir::OpType::DIV:
            if (input_data[0] && input_data[1]) {
                for (int64_t i = 0; i < numel; ++i) {
                    // Security: Check for division by zero
                    if (input_data[1][i] == 0.0f) {
                        // Return IEEE 754 infinity or NaN as appropriate
                        if (input_data[0][i] == 0.0f) {
                            output[i] = std::numeric_limits<float>::quiet_NaN();
                        } else if (input_data[0][i] > 0.0f) {
                            output[i] = std::numeric_limits<float>::infinity();
                        } else {
                            output[i] = -std::numeric_limits<float>::infinity();
                        }
                    } else {
                        output[i] = input_data[0][i] / input_data[1][i];
                    }
                }
            }
            break;

        // Unary ops
        case ir::OpType::NEG:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = -input_data[0][i];
                }
            }
            break;

        case ir::OpType::ABS:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = std::abs(input_data[0][i]);
                }
            }
            break;

        case ir::OpType::SQRT:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = std::sqrt(input_data[0][i]);
                }
            }
            break;

        case ir::OpType::EXP:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = std::exp(input_data[0][i]);
                }
            }
            break;

        case ir::OpType::LOG:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = std::log(input_data[0][i]);
                }
            }
            break;

        case ir::OpType::SIN:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = std::sin(input_data[0][i]);
                }
            }
            break;

        case ir::OpType::COS:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = std::cos(input_data[0][i]);
                }
            }
            break;

        case ir::OpType::TANH:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = std::tanh(input_data[0][i]);
                }
            }
            break;

        // Activation functions
        case ir::OpType::RELU:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = std::max(0.0f, input_data[0][i]);
                }
            }
            break;

        case ir::OpType::SIGMOID:
            if (input_data[0]) {
                for (int64_t i = 0; i < numel; ++i) {
                    output[i] = 1.0f / (1.0f + std::exp(-input_data[0][i]));
                }
            }
            break;

        case ir::OpType::GELU:
            if (input_data[0]) {
                // Approximate GELU: x * sigmoid(1.702 * x)
                for (int64_t i = 0; i < numel; ++i) {
                    float x = input_data[0][i];
                    output[i] = x * (1.0f / (1.0f + std::exp(-1.702f * x)));
                }
            }
            break;

        case ir::OpType::SILU:
            if (input_data[0]) {
                // SiLU: x * sigmoid(x)
                for (int64_t i = 0; i < numel; ++i) {
                    float x = input_data[0][i];
                    output[i] = x / (1.0f + std::exp(-x));
                }
            }
            break;

        // Reductions
        case ir::OpType::SUM: {
            if (input_data[0]) {
                int64_t input_numel = inputs[0]->numel();
                float sum = 0.0f;
                for (int64_t i = 0; i < input_numel; ++i) {
                    sum += input_data[0][i];
                }
                output[0] = sum;
            }
            break;
        }

        case ir::OpType::MEAN: {
            if (input_data[0]) {
                int64_t input_numel = inputs[0]->numel();
                // Security: Check for division by zero (empty tensor)
                if (input_numel == 0) {
                    output[0] = std::numeric_limits<float>::quiet_NaN();
                } else {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < input_numel; ++i) {
                        sum += input_data[0][i];
                    }
                    output[0] = sum / static_cast<float>(input_numel);
                }
            }
            break;
        }

        case ir::OpType::MAX: {
            if (input_data[0]) {
                int64_t input_numel = inputs[0]->numel();
                float max_val = input_data[0][0];
                for (int64_t i = 1; i < input_numel; ++i) {
                    max_val = std::max(max_val, input_data[0][i]);
                }
                output[0] = max_val;
            }
            break;
        }

        case ir::OpType::MIN: {
            if (input_data[0]) {
                int64_t input_numel = inputs[0]->numel();
                float min_val = input_data[0][0];
                for (int64_t i = 1; i < input_numel; ++i) {
                    min_val = std::min(min_val, input_data[0][i]);
                }
                output[0] = min_val;
            }
            break;
        }

        case ir::OpType::SOFTMAX: {
            if (input_data[0]) {
                const auto& shape = inputs[0]->output_spec().shape;
                int ndim = static_cast<int>(shape.size());
                int dim = node_->get_attr<int>("dim", -1);
                if (dim < 0) dim += ndim;

                // Calculate strides for the softmax dimension
                int64_t outer_size = 1;
                int64_t dim_size = shape[dim];
                int64_t inner_size = 1;
                for (int i = 0; i < dim; ++i) outer_size *= shape[i];
                for (int i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

                // Apply softmax along the specified dimension
                for (int64_t outer = 0; outer < outer_size; ++outer) {
                    for (int64_t inner = 0; inner < inner_size; ++inner) {
                        // Find max for numerical stability
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int64_t d = 0; d < dim_size; ++d) {
                            int64_t idx = outer * dim_size * inner_size + d * inner_size + inner;
                            max_val = std::max(max_val, input_data[0][idx]);
                        }

                        // Compute exp(x - max) and sum
                        float sum = 0.0f;
                        for (int64_t d = 0; d < dim_size; ++d) {
                            int64_t idx = outer * dim_size * inner_size + d * inner_size + inner;
                            output[idx] = std::exp(input_data[0][idx] - max_val);
                            sum += output[idx];
                        }

                        // Normalize
                        for (int64_t d = 0; d < dim_size; ++d) {
                            int64_t idx = outer * dim_size * inner_size + d * inner_size + inner;
                            output[idx] /= sum;
                        }
                    }
                }
            }
            break;
        }

        case ir::OpType::LOG_SOFTMAX: {
            if (input_data[0]) {
                const auto& shape = inputs[0]->output_spec().shape;
                int ndim = static_cast<int>(shape.size());
                int dim = node_->get_attr<int>("dim", -1);
                if (dim < 0) dim += ndim;

                // Calculate strides for the softmax dimension
                int64_t outer_size = 1;
                int64_t dim_size = shape[dim];
                int64_t inner_size = 1;
                for (int i = 0; i < dim; ++i) outer_size *= shape[i];
                for (int i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

                // Apply log-softmax using log-sum-exp trick
                for (int64_t outer = 0; outer < outer_size; ++outer) {
                    for (int64_t inner = 0; inner < inner_size; ++inner) {
                        // Find max for numerical stability
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int64_t d = 0; d < dim_size; ++d) {
                            int64_t idx = outer * dim_size * inner_size + d * inner_size + inner;
                            max_val = std::max(max_val, input_data[0][idx]);
                        }

                        // Compute log-sum-exp
                        float log_sum_exp = 0.0f;
                        for (int64_t d = 0; d < dim_size; ++d) {
                            int64_t idx = outer * dim_size * inner_size + d * inner_size + inner;
                            log_sum_exp += std::exp(input_data[0][idx] - max_val);
                        }
                        log_sum_exp = max_val + std::log(log_sum_exp);

                        // Compute log-softmax: x - log_sum_exp
                        for (int64_t d = 0; d < dim_size; ++d) {
                            int64_t idx = outer * dim_size * inner_size + d * inner_size + inner;
                            output[idx] = input_data[0][idx] - log_sum_exp;
                        }
                    }
                }
            }
            break;
        }

        case ir::OpType::CROSS_ENTROPY_LOSS: {
            // Cross-entropy loss: combines log_softmax and nll_loss
            // Input 0: logits (N, C) or (N, C, ...) - predictions
            // Input 1: targets (N,) or (N, ...) - class indices
            if (input_data[0] && input_data[1]) {
                const auto& logits_shape = inputs[0]->output_spec().shape;
                int64_t batch_size = logits_shape[0];
                int64_t num_classes = logits_shape[1];

                // Get optional attributes
                int64_t ignore_index = node_->get_attr<int64_t>("ignore_index", -100);
                float label_smoothing = node_->get_attr<float>("label_smoothing", 0.0f);

                float total_loss = 0.0f;
                int64_t valid_count = 0;

                for (int64_t n = 0; n < batch_size; ++n) {
                    int64_t target = static_cast<int64_t>(input_data[1][n]);

                    // Skip ignored indices
                    if (target == ignore_index) {
                        continue;
                    }

                    // Compute log-softmax for this sample
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int64_t c = 0; c < num_classes; ++c) {
                        max_val = std::max(max_val, input_data[0][n * num_classes + c]);
                    }

                    float log_sum_exp = 0.0f;
                    for (int64_t c = 0; c < num_classes; ++c) {
                        log_sum_exp += std::exp(input_data[0][n * num_classes + c] - max_val);
                    }
                    log_sum_exp = max_val + std::log(log_sum_exp);

                    if (label_smoothing > 0.0f) {
                        // With label smoothing: loss = (1 - smooth) * nll + smooth * uniform_loss
                        float nll = log_sum_exp - input_data[0][n * num_classes + target];
                        float smooth_loss = 0.0f;
                        for (int64_t c = 0; c < num_classes; ++c) {
                            smooth_loss += log_sum_exp - input_data[0][n * num_classes + c];
                        }
                        smooth_loss /= static_cast<float>(num_classes);
                        total_loss += (1.0f - label_smoothing) * nll + label_smoothing * smooth_loss;
                    } else {
                        // Standard cross-entropy: -log_softmax[target]
                        float log_softmax_target = input_data[0][n * num_classes + target] - log_sum_exp;
                        total_loss += -log_softmax_target;
                    }
                    valid_count++;
                }

                // Return mean loss
                output[0] = (valid_count > 0) ? total_loss / static_cast<float>(valid_count) : 0.0f;
            }
            break;
        }

        case ir::OpType::SLICE: {
            if (input_data[0]) {
                const auto& input_shape = inputs[0]->output_spec().shape;
                int ndim = static_cast<int>(input_shape.size());

                int dim = node_->get_attr<int>("dim", 0);
                int64_t start = node_->get_attr<int64_t>("start", 0);
                int64_t end = node_->get_attr<int64_t>("end", input_shape[dim]);

                // Calculate strides for input tensor
                std::vector<int64_t> input_strides(ndim);
                input_strides[ndim - 1] = 1;
                for (int i = ndim - 2; i >= 0; --i) {
                    input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
                }

                // Calculate strides for output tensor
                const auto& output_shape = node_->output_spec().shape;
                std::vector<int64_t> output_strides(ndim);
                output_strides[ndim - 1] = 1;
                for (int i = ndim - 2; i >= 0; --i) {
                    output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
                }

                // Copy sliced data
                std::function<void(int, int64_t, int64_t)> copy_slice = [&](int d, int64_t in_offset, int64_t out_offset) {
                    if (d == ndim) {
                        output[out_offset] = input_data[0][in_offset];
                        return;
                    }

                    int64_t size = (d == dim) ? (end - start) : output_shape[d];
                    int64_t in_start = (d == dim) ? start : 0;

                    for (int64_t i = 0; i < size; ++i) {
                        copy_slice(d + 1,
                                   in_offset + (in_start + i) * input_strides[d],
                                   out_offset + i * output_strides[d]);
                    }
                };

                copy_slice(0, 0, 0);
            }
            break;
        }

        default:
            throw std::runtime_error("Unsupported operation for CPU execution: " +
                                    ir::op_type_name(op));
    }

    node_->mark_evaluated();
}

std::shared_ptr<TensorImpl> TensorImpl::apply_unary(ir::OpType op) const {
    auto result = std::make_shared<TensorImpl>();
    result->graph_ = graph_;

    // Infer output spec
    auto output_spec = ir::infer_unary_op_spec(node_->output_spec(), op);

    // Create operation node
    result->node_ = graph_->create_op(
        op,
        {node_},
        output_spec,
        ir::op_type_name(op) + "_" + std::to_string(graph_->num_nodes())
    );

    return result;
}

std::shared_ptr<TensorImpl> TensorImpl::apply_binary(
    ir::OpType op,
    std::shared_ptr<TensorImpl> other
) const {
    auto result = std::make_shared<TensorImpl>();
    result->graph_ = graph_;

    // Infer output spec
    auto output_spec = ir::infer_binary_op_spec(
        node_->output_spec(),
        other->node_->output_spec(),
        op
    );

    // Create operation node
    result->node_ = graph_->create_op(
        op,
        {node_, other->node_},
        output_spec,
        ir::op_type_name(op) + "_" + std::to_string(graph_->num_nodes())
    );

    return result;
}

std::shared_ptr<TensorImpl> TensorImpl::apply_reduction(
    ir::OpType op,
    std::optional<int> dim,
    bool keepdim
) const {
    auto result = std::make_shared<TensorImpl>();
    result->graph_ = graph_;

    // Infer output spec
    auto output_spec = ir::infer_reduction_spec(node_->output_spec(), dim, keepdim);

    // Create operation node
    auto node = graph_->create_op(
        op,
        {node_},
        output_spec,
        ir::op_type_name(op) + "_" + std::to_string(graph_->num_nodes())
    );

    // Store reduction parameters
    if (dim.has_value()) {
        node->set_attr("dim", dim.value());
    }
    node->set_attr("keepdim", keepdim);

    result->node_ = node;
    return result;
}

std::shared_ptr<TensorImpl> TensorImpl::apply_reshape(
    const std::vector<int64_t>& new_shape
) const {
    auto result = std::make_shared<TensorImpl>();
    result->graph_ = graph_;

    // Infer output spec
    auto output_spec = ir::infer_reshape_spec(node_->output_spec(), new_shape);

    // Create operation node
    auto node = graph_->create_op(
        ir::OpType::RESHAPE,
        {node_},
        output_spec,
        "reshape_" + std::to_string(graph_->num_nodes())
    );

    node->set_attr("new_shape", new_shape);
    result->node_ = node;

    return result;
}

std::shared_ptr<TensorImpl> TensorImpl::apply_transpose(int dim0, int dim1) const {
    auto result = std::make_shared<TensorImpl>();
    result->graph_ = graph_;

    // Infer output spec
    auto output_spec = ir::infer_transpose_spec(node_->output_spec(), dim0, dim1);

    // Create operation node
    auto node = graph_->create_op(
        ir::OpType::TRANSPOSE,
        {node_},
        output_spec,
        "transpose_" + std::to_string(graph_->num_nodes())
    );

    node->set_attr("dim0", dim0);
    node->set_attr("dim1", dim1);
    result->node_ = node;

    return result;
}

}  // namespace pyflame
