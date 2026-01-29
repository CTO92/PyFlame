#include "pyflame/nn/dropout.hpp"
#include "pyflame/ir/op_type.hpp"
#include "pyflame/ir/node.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/core/tensor_impl.hpp"

#include <sstream>
#include <stdexcept>

namespace pyflame::nn {

// ============================================================================
// Dropout Implementation
// ============================================================================

Dropout::Dropout(float p, bool inplace)
    : p_(p), inplace_(inplace)
{
    set_name("Dropout");

    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1]");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    // During eval or p=0, just return input
    if (!is_training() || p_ == 0.0f) {
        return input;
    }

    // During training, apply dropout
    auto graph = input.graph();
    ir::TensorSpec out_spec(input.shape(), input.dtype(), input.layout());

    auto dropout_node = graph->create_op(ir::OpType::DROPOUT, {input.node()}, out_spec);
    dropout_node->set_attr("p", p_);
    dropout_node->set_attr("training", true);
    dropout_node->set_attr("inplace", inplace_);

    auto impl = TensorImpl::from_node(graph, dropout_node);
    return Tensor::from_impl(impl);
}

std::string Dropout::to_string() const {
    std::ostringstream oss;
    oss << "Dropout(p=" << p_;
    if (inplace_) {
        oss << ", inplace=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// Dropout2d Implementation
// ============================================================================

Dropout2d::Dropout2d(float p, bool inplace)
    : p_(p), inplace_(inplace)
{
    set_name("Dropout2d");

    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1]");
    }
}

Tensor Dropout2d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("Dropout2d expects 4D input [N, C, H, W]");
    }

    if (!is_training() || p_ == 0.0f) {
        return input;
    }

    auto graph = input.graph();
    ir::TensorSpec out_spec(shape, input.dtype(), input.layout());

    auto dropout_node = graph->create_op(ir::OpType::DROPOUT, {input.node()}, out_spec);
    dropout_node->set_attr("p", p_);
    dropout_node->set_attr("training", true);
    dropout_node->set_attr("spatial", true);  // Drop entire channels
    dropout_node->set_attr("inplace", inplace_);

    auto impl = TensorImpl::from_node(graph, dropout_node);
    return Tensor::from_impl(impl);
}

std::string Dropout2d::to_string() const {
    std::ostringstream oss;
    oss << "Dropout2d(p=" << p_;
    if (inplace_) {
        oss << ", inplace=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// Dropout1d Implementation
// ============================================================================

Dropout1d::Dropout1d(float p, bool inplace)
    : p_(p), inplace_(inplace)
{
    set_name("Dropout1d");

    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1]");
    }
}

Tensor Dropout1d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("Dropout1d expects 3D input [N, C, L]");
    }

    if (!is_training() || p_ == 0.0f) {
        return input;
    }

    auto graph = input.graph();
    ir::TensorSpec out_spec(shape, input.dtype(), input.layout());

    auto dropout_node = graph->create_op(ir::OpType::DROPOUT, {input.node()}, out_spec);
    dropout_node->set_attr("p", p_);
    dropout_node->set_attr("training", true);
    dropout_node->set_attr("spatial", true);
    dropout_node->set_attr("inplace", inplace_);

    auto impl = TensorImpl::from_node(graph, dropout_node);
    return Tensor::from_impl(impl);
}

std::string Dropout1d::to_string() const {
    std::ostringstream oss;
    oss << "Dropout1d(p=" << p_;
    if (inplace_) {
        oss << ", inplace=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// AlphaDropout Implementation
// ============================================================================

AlphaDropout::AlphaDropout(float p, bool inplace)
    : p_(p), inplace_(inplace)
{
    set_name("AlphaDropout");

    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1]");
    }
}

Tensor AlphaDropout::forward(const Tensor& input) {
    if (!is_training() || p_ == 0.0f) {
        return input;
    }

    // Alpha dropout maintains self-normalizing properties
    // Uses specific alpha and scale values for SELU
    auto graph = input.graph();
    ir::TensorSpec out_spec(input.shape(), input.dtype(), input.layout());

    auto dropout_node = graph->create_op(ir::OpType::DROPOUT, {input.node()}, out_spec);
    dropout_node->set_attr("p", p_);
    dropout_node->set_attr("training", true);
    dropout_node->set_attr("alpha_dropout", true);
    dropout_node->set_attr("inplace", inplace_);

    auto impl = TensorImpl::from_node(graph, dropout_node);
    return Tensor::from_impl(impl);
}

std::string AlphaDropout::to_string() const {
    std::ostringstream oss;
    oss << "AlphaDropout(p=" << p_;
    if (inplace_) {
        oss << ", inplace=True";
    }
    oss << ")";
    return oss.str();
}

}  // namespace pyflame::nn
