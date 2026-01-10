#pragma once

#include "pyflame/nn/module.hpp"

#include <array>
#include <vector>

namespace pyflame::nn {

/// 2D Max Pooling
class MaxPool2d : public Module {
public:
    /// Create a max pooling layer
    MaxPool2d(std::array<int64_t, 2> kernel_size,
              std::array<int64_t, 2> stride = {0, 0},  // 0 = same as kernel_size
              std::array<int64_t, 2> padding = {0, 0});

    /// Convenience constructor with single int
    MaxPool2d(int64_t kernel_size,
              int64_t stride = 0,
              int64_t padding = 0);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

private:
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;

    int64_t calc_output_size(int64_t input_size, int64_t kernel, int64_t stride, int64_t padding) const;
};

/// 2D Average Pooling
class AvgPool2d : public Module {
public:
    AvgPool2d(std::array<int64_t, 2> kernel_size,
              std::array<int64_t, 2> stride = {0, 0},
              std::array<int64_t, 2> padding = {0, 0});

    AvgPool2d(int64_t kernel_size,
              int64_t stride = 0,
              int64_t padding = 0);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

private:
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
};

/// Adaptive Average Pooling (output size is fixed)
class AdaptiveAvgPool2d : public Module {
public:
    /// Create adaptive average pooling
    /// @param output_size Target output size [H, W]
    AdaptiveAvgPool2d(std::array<int64_t, 2> output_size);

    /// Convenience for square output
    AdaptiveAvgPool2d(int64_t output_size);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

private:
    std::array<int64_t, 2> output_size_;
};

/// 1D Max Pooling
class MaxPool1d : public Module {
public:
    MaxPool1d(int64_t kernel_size,
              int64_t stride = 0,
              int64_t padding = 0);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

/// 1D Average Pooling
class AvgPool1d : public Module {
public:
    AvgPool1d(int64_t kernel_size,
              int64_t stride = 0,
              int64_t padding = 0);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
};

/// Adaptive Average Pooling 1D
class AdaptiveAvgPool1d : public Module {
public:
    AdaptiveAvgPool1d(int64_t output_size);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

private:
    int64_t output_size_;
};

/// Global Average Pooling (reduces spatial dims to 1)
class GlobalAvgPool2d : public Module {
public:
    GlobalAvgPool2d() { set_name("GlobalAvgPool2d"); }

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override { return "GlobalAvgPool2d()"; }
};

}  // namespace pyflame::nn
