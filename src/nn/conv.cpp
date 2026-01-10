#include "pyflame/nn/conv.hpp"
#include "pyflame/ir/op_type.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace pyflame::nn {

// ============================================================================
// Conv2d Implementation
// ============================================================================

Conv2d::Conv2d(int64_t in_channels,
               int64_t out_channels,
               std::array<int64_t, 2> kernel_size,
               std::array<int64_t, 2> stride,
               std::array<int64_t, 2> padding,
               std::array<int64_t, 2> dilation,
               int64_t groups,
               bool bias)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , dilation_(dilation)
    , groups_(groups)
    , use_bias_(bias)
{
    set_name("Conv2d");

    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    // Kaiming initialization
    int64_t fan_in = (in_channels / groups) * kernel_size[0] * kernel_size[1];
    float std = std::sqrt(2.0f / static_cast<float>(fan_in));

    // Weight: [out_channels, in_channels/groups, kH, kW]
    weight_ = Tensor::randn({out_channels, in_channels / groups,
                            kernel_size[0], kernel_size[1]}) * std;
    register_parameter("weight", weight_);

    if (use_bias_) {
        bias_ = Tensor::zeros({out_channels});
        register_parameter("bias", bias_);
    }
}

Conv2d::Conv2d(int64_t in_channels,
               int64_t out_channels,
               int64_t kernel_size,
               int64_t stride,
               int64_t padding,
               int64_t dilation,
               int64_t groups,
               bool bias)
    : Conv2d(in_channels, out_channels,
             {kernel_size, kernel_size},
             {stride, stride},
             {padding, padding},
             {dilation, dilation},
             groups, bias)
{
}

int64_t Conv2d::calc_output_size(int64_t input_size, int64_t kernel, int64_t stride,
                                  int64_t padding, int64_t dilation) const {
    return (input_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

Tensor Conv2d::forward(const Tensor& input) {
    // input: [N, C_in, H, W]
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("Conv2d expects 4D input [N, C, H, W]");
    }

    int64_t batch_size = shape[0];
    int64_t in_c = shape[1];
    int64_t in_h = shape[2];
    int64_t in_w = shape[3];

    if (in_c != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    // Calculate output size
    int64_t out_h = calc_output_size(in_h, kernel_size_[0], stride_[0],
                                      padding_[0], dilation_[0]);
    int64_t out_w = calc_output_size(in_w, kernel_size_[1], stride_[1],
                                      padding_[1], dilation_[1]);

    // Create conv2d operation in the graph
    auto input_node = input.node();
    auto weight_node = weight_.node();
    auto graph = input.graph();

    ir::TensorSpec out_spec({batch_size, out_channels_, out_h, out_w},
                           input.dtype(), input.layout());

    auto conv_node = graph->create_op(ir::OpType::CONV2D,
                                      {input_node, weight_node}, out_spec);

    // Store convolution parameters as attributes
    conv_node->set_attr("stride", std::vector<int64_t>{stride_[0], stride_[1]});
    conv_node->set_attr("padding", std::vector<int64_t>{padding_[0], padding_[1]});
    conv_node->set_attr("dilation", std::vector<int64_t>{dilation_[0], dilation_[1]});
    conv_node->set_attr("groups", groups_);

    // Create output tensor from node
    auto impl = TensorImpl::from_node(graph, conv_node);
    Tensor output(impl);

    // Add bias if present
    if (use_bias_) {
        // Reshape bias for broadcasting: [C] -> [1, C, 1, 1]
        auto bias_reshaped = bias_.reshape({1, out_channels_, 1, 1});
        output = output + bias_reshaped;
    }

    return output;
}

std::string Conv2d::to_string() const {
    std::ostringstream oss;
    oss << "Conv2d(" << in_channels_ << ", " << out_channels_
        << ", kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")"
        << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";

    if (padding_[0] != 0 || padding_[1] != 0) {
        oss << ", padding=(" << padding_[0] << ", " << padding_[1] << ")";
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
        oss << ", dilation=(" << dilation_[0] << ", " << dilation_[1] << ")";
    }
    if (groups_ != 1) {
        oss << ", groups=" << groups_;
    }
    if (!use_bias_) {
        oss << ", bias=False";
    }

    oss << ")";
    return oss.str();
}

// ============================================================================
// Conv1d Implementation
// ============================================================================

Conv1d::Conv1d(int64_t in_channels,
               int64_t out_channels,
               int64_t kernel_size,
               int64_t stride,
               int64_t padding,
               int64_t dilation,
               int64_t groups,
               bool bias)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , dilation_(dilation)
    , groups_(groups)
    , use_bias_(bias)
{
    set_name("Conv1d");

    if (in_channels % groups != 0) {
        throw std::invalid_argument("in_channels must be divisible by groups");
    }
    if (out_channels % groups != 0) {
        throw std::invalid_argument("out_channels must be divisible by groups");
    }

    int64_t fan_in = (in_channels / groups) * kernel_size;
    float std = std::sqrt(2.0f / static_cast<float>(fan_in));

    weight_ = Tensor::randn({out_channels, in_channels / groups, kernel_size}) * std;
    register_parameter("weight", weight_);

    if (use_bias_) {
        bias_ = Tensor::zeros({out_channels});
        register_parameter("bias", bias_);
    }
}

Tensor Conv1d::forward(const Tensor& input) {
    // input: [N, C_in, L]
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("Conv1d expects 3D input [N, C, L]");
    }

    int64_t batch_size = shape[0];
    int64_t in_c = shape[1];
    int64_t in_l = shape[2];

    if (in_c != in_channels_) {
        throw std::invalid_argument("Input channels mismatch");
    }

    // Calculate output length
    int64_t out_l = (in_l + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;

    auto input_node = input.node();
    auto weight_node = weight_.node();
    auto graph = input.graph();

    ir::TensorSpec out_spec({batch_size, out_channels_, out_l},
                           input.dtype(), input.layout());

    auto conv_node = graph->create_op(ir::OpType::CONV1D,
                                      {input_node, weight_node}, out_spec);

    conv_node->set_attr("stride", stride_);
    conv_node->set_attr("padding", padding_);
    conv_node->set_attr("dilation", dilation_);
    conv_node->set_attr("groups", groups_);

    auto impl = TensorImpl::from_node(graph, conv_node);
    Tensor output(impl);

    if (use_bias_) {
        auto bias_reshaped = bias_.reshape({1, out_channels_, 1});
        output = output + bias_reshaped;
    }

    return output;
}

std::string Conv1d::to_string() const {
    std::ostringstream oss;
    oss << "Conv1d(" << in_channels_ << ", " << out_channels_
        << ", kernel_size=" << kernel_size_
        << ", stride=" << stride_;

    if (padding_ != 0) {
        oss << ", padding=" << padding_;
    }
    if (dilation_ != 1) {
        oss << ", dilation=" << dilation_;
    }
    if (groups_ != 1) {
        oss << ", groups=" << groups_;
    }
    if (!use_bias_) {
        oss << ", bias=False";
    }

    oss << ")";
    return oss.str();
}

}  // namespace pyflame::nn
