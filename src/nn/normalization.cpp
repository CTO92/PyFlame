#include "pyflame/nn/normalization.hpp"
#include "pyflame/ir/op_type.hpp"
#include "pyflame/ir/node.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/core/tensor_impl.hpp"

#include <sstream>
#include <stdexcept>
#include <numeric>

namespace pyflame::nn {

// ============================================================================
// BatchNorm2d Implementation
// ============================================================================

BatchNorm2d::BatchNorm2d(int64_t num_features,
                         float eps,
                         float momentum,
                         bool affine,
                         bool track_running_stats)
    : num_features_(num_features)
    , eps_(eps)
    , momentum_(momentum)
    , affine_(affine)
    , track_running_stats_(track_running_stats)
{
    set_name("BatchNorm2d");

    if (affine_) {
        weight_ = Tensor::ones({num_features});
        bias_ = Tensor::zeros({num_features});
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }

    if (track_running_stats_) {
        running_mean_ = Tensor::zeros({num_features});
        running_var_ = Tensor::ones({num_features});
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
    }
}

void BatchNorm2d::reset_running_stats() {
    if (track_running_stats_) {
        running_mean_ = Tensor::zeros({num_features_});
        running_var_ = Tensor::ones({num_features_});
        num_batches_tracked_ = 0;
    }
}

Tensor BatchNorm2d::forward(const Tensor& input) {
    // input: [N, C, H, W]
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("BatchNorm2d expects 4D input [N, C, H, W]");
    }

    if (shape[1] != num_features_) {
        throw std::invalid_argument("Number of channels mismatch");
    }

    auto input_node = input.node();
    auto graph = input.graph();

    if (is_training() && track_running_stats_) {
        num_batches_tracked_++;
    }

    // Create batch norm operation in graph
    ir::TensorSpec out_spec(shape, input.dtype(), input.layout());

    std::vector<std::shared_ptr<ir::Node>> inputs_vec = {input_node};
    if (affine_) {
        inputs_vec.push_back(weight_.node());
        inputs_vec.push_back(bias_.node());
    }
    if (track_running_stats_) {
        inputs_vec.push_back(running_mean_.node());
        inputs_vec.push_back(running_var_.node());
    }

    auto bn_node = graph->create_op(ir::OpType::BATCH_NORM, inputs_vec, out_spec);
    bn_node->set_attr("eps", eps_);
    bn_node->set_attr("momentum", momentum_);
    bn_node->set_attr("training", is_training());
    bn_node->set_attr("affine", affine_);
    bn_node->set_attr("track_running_stats", track_running_stats_);

    auto impl = TensorImpl::from_node(graph, bn_node);
    return Tensor::from_impl(impl);
}

std::string BatchNorm2d::to_string() const {
    std::ostringstream oss;
    oss << "BatchNorm2d(" << num_features_
        << ", eps=" << eps_
        << ", momentum=" << momentum_
        << ", affine=" << (affine_ ? "True" : "False")
        << ", track_running_stats=" << (track_running_stats_ ? "True" : "False")
        << ")";
    return oss.str();
}

// ============================================================================
// BatchNorm1d Implementation
// ============================================================================

BatchNorm1d::BatchNorm1d(int64_t num_features,
                         float eps,
                         float momentum,
                         bool affine,
                         bool track_running_stats)
    : num_features_(num_features)
    , eps_(eps)
    , momentum_(momentum)
    , affine_(affine)
    , track_running_stats_(track_running_stats)
{
    set_name("BatchNorm1d");

    if (affine_) {
        weight_ = Tensor::ones({num_features});
        bias_ = Tensor::zeros({num_features});
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }

    if (track_running_stats_) {
        running_mean_ = Tensor::zeros({num_features});
        running_var_ = Tensor::ones({num_features});
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
    }
}

void BatchNorm1d::reset_running_stats() {
    if (track_running_stats_) {
        running_mean_ = Tensor::zeros({num_features_});
        running_var_ = Tensor::ones({num_features_});
        num_batches_tracked_ = 0;
    }
}

Tensor BatchNorm1d::forward(const Tensor& input) {
    // input: [N, C] or [N, C, L]
    auto shape = input.shape();
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::invalid_argument("BatchNorm1d expects 2D or 3D input");
    }

    if (shape[1] != num_features_) {
        throw std::invalid_argument("Number of features mismatch");
    }

    auto input_node = input.node();
    auto graph = input.graph();

    if (is_training() && track_running_stats_) {
        num_batches_tracked_++;
    }

    ir::TensorSpec out_spec(shape, input.dtype(), input.layout());

    std::vector<std::shared_ptr<ir::Node>> inputs_vec = {input_node};
    if (affine_) {
        inputs_vec.push_back(weight_.node());
        inputs_vec.push_back(bias_.node());
    }
    if (track_running_stats_) {
        inputs_vec.push_back(running_mean_.node());
        inputs_vec.push_back(running_var_.node());
    }

    auto bn_node = graph->create_op(ir::OpType::BATCH_NORM, inputs_vec, out_spec);
    bn_node->set_attr("eps", eps_);
    bn_node->set_attr("momentum", momentum_);
    bn_node->set_attr("training", is_training());
    bn_node->set_attr("affine", affine_);
    bn_node->set_attr("track_running_stats", track_running_stats_);

    auto impl = TensorImpl::from_node(graph, bn_node);
    return Tensor::from_impl(impl);
}

std::string BatchNorm1d::to_string() const {
    std::ostringstream oss;
    oss << "BatchNorm1d(" << num_features_
        << ", eps=" << eps_
        << ", momentum=" << momentum_
        << ")";
    return oss.str();
}

// ============================================================================
// LayerNorm Implementation
// ============================================================================

LayerNorm::LayerNorm(std::vector<int64_t> normalized_shape,
                     float eps,
                     bool elementwise_affine)
    : normalized_shape_(std::move(normalized_shape))
    , eps_(eps)
    , elementwise_affine_(elementwise_affine)
{
    set_name("LayerNorm");

    if (elementwise_affine_) {
        weight_ = Tensor::ones(normalized_shape_);
        bias_ = Tensor::zeros(normalized_shape_);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }
}

LayerNorm::LayerNorm(int64_t normalized_shape,
                     float eps,
                     bool elementwise_affine)
    : LayerNorm(std::vector<int64_t>{normalized_shape}, eps, elementwise_affine)
{
}

Tensor LayerNorm::forward(const Tensor& input) {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t norm_dims = static_cast<int64_t>(normalized_shape_.size());

    // Check that last norm_dims dimensions match normalized_shape
    for (int64_t i = 0; i < norm_dims; ++i) {
        int64_t idx = ndim - norm_dims + i;
        if (shape[idx] != normalized_shape_[i]) {
            throw std::invalid_argument("Input shape doesn't match normalized_shape");
        }
    }

    auto input_node = input.node();
    auto graph = input.graph();

    ir::TensorSpec out_spec(shape, input.dtype(), input.layout());

    std::vector<std::shared_ptr<ir::Node>> inputs_vec = {input_node};
    if (elementwise_affine_) {
        inputs_vec.push_back(weight_.node());
        inputs_vec.push_back(bias_.node());
    }

    auto ln_node = graph->create_op(ir::OpType::LAYER_NORM, inputs_vec, out_spec);
    ln_node->set_attr("normalized_shape", normalized_shape_);
    ln_node->set_attr("eps", eps_);
    ln_node->set_attr("elementwise_affine", elementwise_affine_);

    auto impl = TensorImpl::from_node(graph, ln_node);
    return Tensor::from_impl(impl);
}

std::string LayerNorm::to_string() const {
    std::ostringstream oss;
    oss << "LayerNorm([";
    for (size_t i = 0; i < normalized_shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << normalized_shape_[i];
    }
    oss << "], eps=" << eps_
        << ", elementwise_affine=" << (elementwise_affine_ ? "True" : "False")
        << ")";
    return oss.str();
}

// ============================================================================
// GroupNorm Implementation
// ============================================================================

GroupNorm::GroupNorm(int64_t num_groups,
                     int64_t num_channels,
                     float eps,
                     bool affine)
    : num_groups_(num_groups)
    , num_channels_(num_channels)
    , eps_(eps)
    , affine_(affine)
{
    set_name("GroupNorm");

    if (num_channels % num_groups != 0) {
        throw std::invalid_argument("num_channels must be divisible by num_groups");
    }

    if (affine_) {
        weight_ = Tensor::ones({num_channels});
        bias_ = Tensor::zeros({num_channels});
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }
}

Tensor GroupNorm::forward(const Tensor& input) {
    // input: [N, C, ...]
    auto shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument("GroupNorm expects at least 2D input");
    }

    if (shape[1] != num_channels_) {
        throw std::invalid_argument("Number of channels mismatch");
    }

    auto input_node = input.node();
    auto graph = input.graph();

    ir::TensorSpec out_spec(shape, input.dtype(), input.layout());

    std::vector<std::shared_ptr<ir::Node>> inputs_vec = {input_node};
    if (affine_) {
        inputs_vec.push_back(weight_.node());
        inputs_vec.push_back(bias_.node());
    }

    auto gn_node = graph->create_op(ir::OpType::GROUP_NORM, inputs_vec, out_spec);
    gn_node->set_attr("num_groups", num_groups_);
    gn_node->set_attr("eps", eps_);
    gn_node->set_attr("affine", affine_);

    auto impl = TensorImpl::from_node(graph, gn_node);
    return Tensor::from_impl(impl);
}

std::string GroupNorm::to_string() const {
    std::ostringstream oss;
    oss << "GroupNorm(" << num_groups_ << ", " << num_channels_
        << ", eps=" << eps_
        << ", affine=" << (affine_ ? "True" : "False")
        << ")";
    return oss.str();
}

}  // namespace pyflame::nn
