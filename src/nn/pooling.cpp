#include "pyflame/nn/pooling.hpp"
#include "pyflame/ir/op_type.hpp"
#include "pyflame/ir/node.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/core/tensor_impl.hpp"

#include <sstream>
#include <stdexcept>

namespace pyflame::nn {

// ============================================================================
// MaxPool2d Implementation
// ============================================================================

MaxPool2d::MaxPool2d(std::array<int64_t, 2> kernel_size,
                     std::array<int64_t, 2> stride,
                     std::array<int64_t, 2> padding)
    : kernel_size_(kernel_size)
    , stride_(stride[0] == 0 ? kernel_size : stride)
    , padding_(padding)
{
    set_name("MaxPool2d");
}

MaxPool2d::MaxPool2d(int64_t kernel_size, int64_t stride, int64_t padding)
    : MaxPool2d({kernel_size, kernel_size},
                {stride == 0 ? kernel_size : stride, stride == 0 ? kernel_size : stride},
                {padding, padding})
{
}

int64_t MaxPool2d::calc_output_size(int64_t input_size, int64_t kernel,
                                     int64_t stride, int64_t padding) const {
    return (input_size + 2 * padding - kernel) / stride + 1;
}

Tensor MaxPool2d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d expects 4D input [N, C, H, W]");
    }

    int64_t batch_size = shape[0];
    int64_t channels = shape[1];
    int64_t in_h = shape[2];
    int64_t in_w = shape[3];

    int64_t out_h = calc_output_size(in_h, kernel_size_[0], stride_[0], padding_[0]);
    int64_t out_w = calc_output_size(in_w, kernel_size_[1], stride_[1], padding_[1]);

    auto graph = input.graph();
    ir::TensorSpec out_spec({batch_size, channels, out_h, out_w},
                           input.dtype(), input.layout());

    auto pool_node = graph->create_op(ir::OpType::MAX_POOL2D, {input.node()}, out_spec);
    pool_node->set_attr("kernel_size", std::vector<int64_t>{kernel_size_[0], kernel_size_[1]});
    pool_node->set_attr("stride", std::vector<int64_t>{stride_[0], stride_[1]});
    pool_node->set_attr("padding", std::vector<int64_t>{padding_[0], padding_[1]});

    auto impl = TensorImpl::from_node(graph, pool_node);
    return Tensor::from_impl(impl);
}

std::string MaxPool2d::to_string() const {
    std::ostringstream oss;
    oss << "MaxPool2d(kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")"
        << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";
    if (padding_[0] != 0 || padding_[1] != 0) {
        oss << ", padding=(" << padding_[0] << ", " << padding_[1] << ")";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// AvgPool2d Implementation
// ============================================================================

AvgPool2d::AvgPool2d(std::array<int64_t, 2> kernel_size,
                     std::array<int64_t, 2> stride,
                     std::array<int64_t, 2> padding)
    : kernel_size_(kernel_size)
    , stride_(stride[0] == 0 ? kernel_size : stride)
    , padding_(padding)
{
    set_name("AvgPool2d");
}

AvgPool2d::AvgPool2d(int64_t kernel_size, int64_t stride, int64_t padding)
    : AvgPool2d({kernel_size, kernel_size},
                {stride == 0 ? kernel_size : stride, stride == 0 ? kernel_size : stride},
                {padding, padding})
{
}

Tensor AvgPool2d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d expects 4D input [N, C, H, W]");
    }

    int64_t batch_size = shape[0];
    int64_t channels = shape[1];
    int64_t in_h = shape[2];
    int64_t in_w = shape[3];

    int64_t out_h = (in_h + 2 * padding_[0] - kernel_size_[0]) / stride_[0] + 1;
    int64_t out_w = (in_w + 2 * padding_[1] - kernel_size_[1]) / stride_[1] + 1;

    auto graph = input.graph();
    ir::TensorSpec out_spec({batch_size, channels, out_h, out_w},
                           input.dtype(), input.layout());

    auto pool_node = graph->create_op(ir::OpType::AVG_POOL2D, {input.node()}, out_spec);
    pool_node->set_attr("kernel_size", std::vector<int64_t>{kernel_size_[0], kernel_size_[1]});
    pool_node->set_attr("stride", std::vector<int64_t>{stride_[0], stride_[1]});
    pool_node->set_attr("padding", std::vector<int64_t>{padding_[0], padding_[1]});

    auto impl = TensorImpl::from_node(graph, pool_node);
    return Tensor::from_impl(impl);
}

std::string AvgPool2d::to_string() const {
    std::ostringstream oss;
    oss << "AvgPool2d(kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")"
        << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";
    if (padding_[0] != 0 || padding_[1] != 0) {
        oss << ", padding=(" << padding_[0] << ", " << padding_[1] << ")";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// AdaptiveAvgPool2d Implementation
// ============================================================================

AdaptiveAvgPool2d::AdaptiveAvgPool2d(std::array<int64_t, 2> output_size)
    : output_size_(output_size)
{
    set_name("AdaptiveAvgPool2d");
}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(int64_t output_size)
    : AdaptiveAvgPool2d({output_size, output_size})
{
}

Tensor AdaptiveAvgPool2d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AdaptiveAvgPool2d expects 4D input [N, C, H, W]");
    }

    int64_t batch_size = shape[0];
    int64_t channels = shape[1];

    auto graph = input.graph();
    ir::TensorSpec out_spec({batch_size, channels, output_size_[0], output_size_[1]},
                           input.dtype(), input.layout());

    auto pool_node = graph->create_op(ir::OpType::ADAPTIVE_AVG_POOL2D, {input.node()}, out_spec);
    pool_node->set_attr("output_size", std::vector<int64_t>{output_size_[0], output_size_[1]});

    auto impl = TensorImpl::from_node(graph, pool_node);
    return Tensor::from_impl(impl);
}

std::string AdaptiveAvgPool2d::to_string() const {
    std::ostringstream oss;
    oss << "AdaptiveAvgPool2d(output_size=(" << output_size_[0] << ", " << output_size_[1] << "))";
    return oss.str();
}

// ============================================================================
// MaxPool1d Implementation
// ============================================================================

MaxPool1d::MaxPool1d(int64_t kernel_size, int64_t stride, int64_t padding)
    : kernel_size_(kernel_size)
    , stride_(stride == 0 ? kernel_size : stride)
    , padding_(padding)
{
    set_name("MaxPool1d");
}

Tensor MaxPool1d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("MaxPool1d expects 3D input [N, C, L]");
    }

    int64_t batch_size = shape[0];
    int64_t channels = shape[1];
    int64_t in_l = shape[2];

    int64_t out_l = (in_l + 2 * padding_ - kernel_size_) / stride_ + 1;

    auto graph = input.graph();
    ir::TensorSpec out_spec({batch_size, channels, out_l}, input.dtype(), input.layout());

    auto pool_node = graph->create_op(ir::OpType::MAX_POOL1D, {input.node()}, out_spec);
    pool_node->set_attr("kernel_size", kernel_size_);
    pool_node->set_attr("stride", stride_);
    pool_node->set_attr("padding", padding_);

    auto impl = TensorImpl::from_node(graph, pool_node);
    return Tensor::from_impl(impl);
}

std::string MaxPool1d::to_string() const {
    std::ostringstream oss;
    oss << "MaxPool1d(kernel_size=" << kernel_size_
        << ", stride=" << stride_;
    if (padding_ != 0) {
        oss << ", padding=" << padding_;
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// AvgPool1d Implementation
// ============================================================================

AvgPool1d::AvgPool1d(int64_t kernel_size, int64_t stride, int64_t padding)
    : kernel_size_(kernel_size)
    , stride_(stride == 0 ? kernel_size : stride)
    , padding_(padding)
{
    set_name("AvgPool1d");
}

Tensor AvgPool1d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("AvgPool1d expects 3D input [N, C, L]");
    }

    int64_t batch_size = shape[0];
    int64_t channels = shape[1];
    int64_t in_l = shape[2];

    int64_t out_l = (in_l + 2 * padding_ - kernel_size_) / stride_ + 1;

    auto graph = input.graph();
    ir::TensorSpec out_spec({batch_size, channels, out_l}, input.dtype(), input.layout());

    auto pool_node = graph->create_op(ir::OpType::AVG_POOL1D, {input.node()}, out_spec);
    pool_node->set_attr("kernel_size", kernel_size_);
    pool_node->set_attr("stride", stride_);
    pool_node->set_attr("padding", padding_);

    auto impl = TensorImpl::from_node(graph, pool_node);
    return Tensor::from_impl(impl);
}

std::string AvgPool1d::to_string() const {
    std::ostringstream oss;
    oss << "AvgPool1d(kernel_size=" << kernel_size_
        << ", stride=" << stride_;
    if (padding_ != 0) {
        oss << ", padding=" << padding_;
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// AdaptiveAvgPool1d Implementation
// ============================================================================

AdaptiveAvgPool1d::AdaptiveAvgPool1d(int64_t output_size)
    : output_size_(output_size)
{
    set_name("AdaptiveAvgPool1d");
}

Tensor AdaptiveAvgPool1d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("AdaptiveAvgPool1d expects 3D input [N, C, L]");
    }

    int64_t batch_size = shape[0];
    int64_t channels = shape[1];

    auto graph = input.graph();
    ir::TensorSpec out_spec({batch_size, channels, output_size_},
                           input.dtype(), input.layout());

    auto pool_node = graph->create_op(ir::OpType::ADAPTIVE_AVG_POOL1D, {input.node()}, out_spec);
    pool_node->set_attr("output_size", std::vector<int64_t>{output_size_});

    auto impl = TensorImpl::from_node(graph, pool_node);
    return Tensor::from_impl(impl);
}

std::string AdaptiveAvgPool1d::to_string() const {
    std::ostringstream oss;
    oss << "AdaptiveAvgPool1d(output_size=" << output_size_ << ")";
    return oss.str();
}

// ============================================================================
// GlobalAvgPool2d Implementation
// ============================================================================

Tensor GlobalAvgPool2d::forward(const Tensor& input) {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("GlobalAvgPool2d expects 4D input [N, C, H, W]");
    }

    // Use adaptive pool with output size 1x1
    int64_t batch_size = shape[0];
    int64_t channels = shape[1];

    auto graph = input.graph();
    ir::TensorSpec out_spec({batch_size, channels, 1, 1},
                           input.dtype(), input.layout());

    auto pool_node = graph->create_op(ir::OpType::ADAPTIVE_AVG_POOL2D, {input.node()}, out_spec);
    pool_node->set_attr("output_size", std::vector<int64_t>{1, 1});

    auto impl = TensorImpl::from_node(graph, pool_node);
    return Tensor::from_impl(impl);
}

}  // namespace pyflame::nn
