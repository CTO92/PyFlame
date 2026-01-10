#include "pyflame/core/tensor_impl.hpp"
#include "pyflame/ir/shape_inference.hpp"
#include <cmath>
#include <algorithm>

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
                    output[i] = input_data[0][i] / input_data[1][i];
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
                float sum = 0.0f;
                for (int64_t i = 0; i < input_numel; ++i) {
                    sum += input_data[0][i];
                }
                output[0] = sum / static_cast<float>(input_numel);
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
