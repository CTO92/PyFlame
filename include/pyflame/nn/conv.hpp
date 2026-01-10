#pragma once

#include "pyflame/nn/module.hpp"

#include <array>
#include <vector>

namespace pyflame::nn {

/// 2D Convolution layer
/// Applies a 2D convolution over an input signal composed of several input planes.
class Conv2d : public Module {
public:
    /// Create a 2D convolution layer
    /// @param in_channels Number of channels in the input image
    /// @param out_channels Number of channels produced by the convolution
    /// @param kernel_size Size of the convolving kernel
    /// @param stride Stride of the convolution (default: 1)
    /// @param padding Zero-padding added to both sides (default: 0)
    /// @param dilation Spacing between kernel elements (default: 1)
    /// @param groups Number of blocked connections from input to output (default: 1)
    /// @param bias If true, adds a learnable bias (default: true)
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           std::array<int64_t, 2> kernel_size,
           std::array<int64_t, 2> stride = {1, 1},
           std::array<int64_t, 2> padding = {0, 0},
           std::array<int64_t, 2> dilation = {1, 1},
           int64_t groups = 1,
           bool bias = true);

    /// Convenience constructor with single int for kernel size
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           int64_t kernel_size,
           int64_t stride = 1,
           int64_t padding = 0,
           int64_t dilation = 1,
           int64_t groups = 1,
           bool bias = true);

    /// Forward pass
    /// @param input Input tensor of shape [N, C_in, H, W]
    /// @return Output tensor of shape [N, C_out, H_out, W_out]
    Tensor forward(const Tensor& input) override;

    /// String representation
    std::string to_string() const override;

    // Accessors
    int64_t in_channels() const { return in_channels_; }
    int64_t out_channels() const { return out_channels_; }
    std::array<int64_t, 2> kernel_size() const { return kernel_size_; }
    std::array<int64_t, 2> stride() const { return stride_; }
    std::array<int64_t, 2> padding() const { return padding_; }
    std::array<int64_t, 2> dilation() const { return dilation_; }
    int64_t groups() const { return groups_; }
    bool has_bias() const { return use_bias_; }

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }
    Tensor& bias() { return bias_; }
    const Tensor& bias() const { return bias_; }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
    std::array<int64_t, 2> dilation_;
    int64_t groups_;
    bool use_bias_;

    Tensor weight_;  // [out_channels, in_channels/groups, kH, kW]
    Tensor bias_;    // [out_channels]

    /// Calculate output size for one dimension
    int64_t calc_output_size(int64_t input_size, int64_t kernel, int64_t stride,
                             int64_t padding, int64_t dilation) const;
};

/// 1D Convolution layer
class Conv1d : public Module {
public:
    Conv1d(int64_t in_channels,
           int64_t out_channels,
           int64_t kernel_size,
           int64_t stride = 1,
           int64_t padding = 0,
           int64_t dilation = 1,
           int64_t groups = 1,
           bool bias = true);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
    bool use_bias_;

    Tensor weight_;  // [out_channels, in_channels/groups, kernel_size]
    Tensor bias_;    // [out_channels]
};

}  // namespace pyflame::nn
