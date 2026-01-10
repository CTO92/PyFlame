#pragma once

#include "pyflame/nn/module.hpp"

namespace pyflame::nn {

/// Dropout layer
/// During training, randomly zeroes elements with probability p
/// During evaluation, returns input unchanged
class Dropout : public Module {
public:
    /// Create a dropout layer
    /// @param p Probability of element being zeroed (default: 0.5)
    /// @param inplace If true, operates in-place (default: false)
    Dropout(float p = 0.5f, bool inplace = false);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

    float p() const { return p_; }
    bool inplace() const { return inplace_; }

private:
    float p_;
    bool inplace_;
};

/// Dropout2d (spatial dropout)
/// Drops entire channels for 4D input [N, C, H, W]
class Dropout2d : public Module {
public:
    Dropout2d(float p = 0.5f, bool inplace = false);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

    float p() const { return p_; }
    bool inplace() const { return inplace_; }

private:
    float p_;
    bool inplace_;
};

/// Dropout1d (drops entire channels for 3D input)
class Dropout1d : public Module {
public:
    Dropout1d(float p = 0.5f, bool inplace = false);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

    float p() const { return p_; }
    bool inplace() const { return inplace_; }

private:
    float p_;
    bool inplace_;
};

/// Alpha Dropout (for SELU networks)
class AlphaDropout : public Module {
public:
    AlphaDropout(float p = 0.5f, bool inplace = false);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

private:
    float p_;
    bool inplace_;
};

}  // namespace pyflame::nn
