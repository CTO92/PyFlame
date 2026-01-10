#pragma once

#include "pyflame/nn/module.hpp"

#include <vector>

namespace pyflame::nn {

/// Batch Normalization over a 2D or 4D input
/// Applies: y = (x - mean) / sqrt(var + eps) * gamma + beta
class BatchNorm2d : public Module {
public:
    /// Create a batch normalization layer
    /// @param num_features Number of features (C from [N, C, H, W])
    /// @param eps Small constant for numerical stability
    /// @param momentum Momentum for running stats
    /// @param affine If true, learns gamma and beta
    /// @param track_running_stats If true, tracks running mean/var
    BatchNorm2d(int64_t num_features,
                float eps = 1e-5f,
                float momentum = 0.1f,
                bool affine = true,
                bool track_running_stats = true);

    /// Forward pass
    Tensor forward(const Tensor& input) override;

    /// Reset running statistics
    void reset_running_stats();

    /// String representation
    std::string to_string() const override;

    // Accessors
    int64_t num_features() const { return num_features_; }
    float eps() const { return eps_; }
    float momentum() const { return momentum_; }
    bool affine() const { return affine_; }
    bool track_running_stats() const { return track_running_stats_; }

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }
    Tensor& bias() { return bias_; }
    const Tensor& bias() const { return bias_; }
    Tensor& running_mean() { return running_mean_; }
    const Tensor& running_mean() const { return running_mean_; }
    Tensor& running_var() { return running_var_; }
    const Tensor& running_var() const { return running_var_; }

    int64_t num_batches_tracked() const { return num_batches_tracked_; }

private:
    int64_t num_features_;
    float eps_;
    float momentum_;
    bool affine_;
    bool track_running_stats_;

    Tensor weight_;        // gamma: [num_features]
    Tensor bias_;          // beta: [num_features]
    Tensor running_mean_;  // [num_features]
    Tensor running_var_;   // [num_features]
    int64_t num_batches_tracked_ = 0;
};

/// Batch Normalization over a 1D input
class BatchNorm1d : public Module {
public:
    BatchNorm1d(int64_t num_features,
                float eps = 1e-5f,
                float momentum = 0.1f,
                bool affine = true,
                bool track_running_stats = true);

    Tensor forward(const Tensor& input) override;
    void reset_running_stats();
    std::string to_string() const override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
    Tensor& running_mean() { return running_mean_; }
    Tensor& running_var() { return running_var_; }

private:
    int64_t num_features_;
    float eps_;
    float momentum_;
    bool affine_;
    bool track_running_stats_;

    Tensor weight_;
    Tensor bias_;
    Tensor running_mean_;
    Tensor running_var_;
    int64_t num_batches_tracked_ = 0;
};

/// Layer Normalization
/// Normalizes over the last D dimensions
class LayerNorm : public Module {
public:
    /// Create a layer normalization layer
    /// @param normalized_shape Shape of the last D dimensions to normalize
    /// @param eps Small constant for numerical stability
    /// @param elementwise_affine If true, learns gamma and beta
    LayerNorm(std::vector<int64_t> normalized_shape,
              float eps = 1e-5f,
              bool elementwise_affine = true);

    /// Convenience constructor for single dimension
    LayerNorm(int64_t normalized_shape,
              float eps = 1e-5f,
              bool elementwise_affine = true);

    /// Forward pass
    Tensor forward(const Tensor& input) override;

    /// String representation
    std::string to_string() const override;

    // Accessors
    const std::vector<int64_t>& normalized_shape() const { return normalized_shape_; }
    float eps() const { return eps_; }
    bool elementwise_affine() const { return elementwise_affine_; }

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }
    Tensor& bias() { return bias_; }
    const Tensor& bias() const { return bias_; }

private:
    std::vector<int64_t> normalized_shape_;
    float eps_;
    bool elementwise_affine_;

    Tensor weight_;  // gamma
    Tensor bias_;    // beta
};

/// Group Normalization
/// Divides channels into groups and normalizes within each group
class GroupNorm : public Module {
public:
    /// Create a group normalization layer
    /// @param num_groups Number of groups to divide channels into
    /// @param num_channels Number of channels
    /// @param eps Small constant for numerical stability
    /// @param affine If true, learns gamma and beta
    GroupNorm(int64_t num_groups,
              int64_t num_channels,
              float eps = 1e-5f,
              bool affine = true);

    Tensor forward(const Tensor& input) override;
    std::string to_string() const override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }

private:
    int64_t num_groups_;
    int64_t num_channels_;
    float eps_;
    bool affine_;

    Tensor weight_;
    Tensor bias_;
};

}  // namespace pyflame::nn
