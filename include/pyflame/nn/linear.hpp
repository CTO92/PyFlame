#pragma once

#include "pyflame/nn/module.hpp"

namespace pyflame::nn {

/// Linear (fully connected) layer
/// Applies y = x @ weight^T + bias
class Linear : public Module {
public:
    /// Create a linear layer
    /// @param in_features Size of each input sample
    /// @param out_features Size of each output sample
    /// @param bias If true, adds a learnable bias
    Linear(int64_t in_features, int64_t out_features, bool bias = true);

    /// Forward pass
    Tensor forward(const Tensor& input) override;

    /// String representation
    std::string to_string() const override;

    // Accessors
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    bool has_bias() const { return use_bias_; }

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }
    Tensor& bias() { return bias_; }
    const Tensor& bias() const { return bias_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    bool use_bias_;

    Tensor weight_;  // [out_features, in_features]
    Tensor bias_;    // [out_features]
};

}  // namespace pyflame::nn
