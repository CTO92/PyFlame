#pragma once

#include "pyflame/nn/module.hpp"
#include "pyflame/nn/linear.hpp"
#include "pyflame/nn/dropout.hpp"

#include <tuple>

namespace pyflame::nn {

/// Multi-Head Attention
/// Allows the model to attend to information from different representation
/// subspaces at different positions.
class MultiheadAttention : public Module {
public:
    /// Create a multi-head attention layer
    /// @param embed_dim Total dimension of the model
    /// @param num_heads Number of parallel attention heads
    /// @param dropout Dropout probability on attention weights
    /// @param bias If true, adds bias to input/output projections
    /// @param add_bias_kv If true, adds bias to key and value sequences
    /// @param add_zero_attn If true, adds a new batch of zeros to key and value
    /// @param kdim Total dimension of key (default: embed_dim)
    /// @param vdim Total dimension of value (default: embed_dim)
    /// @param batch_first If true, input is [batch, seq, feature]
    MultiheadAttention(int64_t embed_dim,
                       int64_t num_heads,
                       float dropout = 0.0f,
                       bool bias = true,
                       bool add_bias_kv = false,
                       bool add_zero_attn = false,
                       int64_t kdim = 0,
                       int64_t vdim = 0,
                       bool batch_first = false);

    /// Forward pass
    /// @param query Query tensor [L, N, E] or [N, L, E] if batch_first
    /// @param key Key tensor [S, N, E] or [N, S, E] if batch_first
    /// @param value Value tensor [S, N, E] or [N, S, E] if batch_first
    /// @param attn_mask Optional attention mask
    /// @param need_weights If true, returns attention weights
    /// @return Tuple of (attention output, attention weights or empty)
    std::tuple<Tensor, Tensor> forward(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        const Tensor& attn_mask = Tensor(),
        bool need_weights = true
    );

    /// Single-input forward (uses input as query, key, value)
    Tensor forward(const Tensor& input) override;

    std::string to_string() const override;

    // Accessors
    int64_t embed_dim() const { return embed_dim_; }
    int64_t num_heads() const { return num_heads_; }
    int64_t head_dim() const { return head_dim_; }
    float dropout_p() const { return dropout_p_; }
    bool batch_first() const { return batch_first_; }

private:
    int64_t embed_dim_;
    int64_t num_heads_;
    int64_t head_dim_;
    int64_t kdim_;
    int64_t vdim_;
    float dropout_p_;
    bool bias_;
    bool add_bias_kv_;
    bool add_zero_attn_;
    bool batch_first_;

    // Projection layers
    std::shared_ptr<Linear> q_proj_;
    std::shared_ptr<Linear> k_proj_;
    std::shared_ptr<Linear> v_proj_;
    std::shared_ptr<Linear> out_proj_;

    std::shared_ptr<Dropout> dropout_;

    // Optional bias for key/value
    Tensor bias_k_;
    Tensor bias_v_;

    /// Compute scaled dot-product attention
    std::tuple<Tensor, Tensor> scaled_dot_product_attention(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        const Tensor& attn_mask
    );
};

/// Self-Attention layer (simplified version)
/// Query, Key, Value all come from the same input
class SelfAttention : public Module {
public:
    SelfAttention(int64_t embed_dim,
                  int64_t num_heads,
                  float dropout = 0.0f,
                  bool bias = true);

    Tensor forward(const Tensor& input) override;

    std::string to_string() const override;

private:
    std::shared_ptr<MultiheadAttention> mha_;
};

/// Cross-Attention layer
/// Query comes from one sequence, Key/Value from another
class CrossAttention : public Module {
public:
    CrossAttention(int64_t embed_dim,
                   int64_t num_heads,
                   float dropout = 0.0f,
                   bool bias = true);

    /// Forward with separate query and context
    Tensor forward(const Tensor& query, const Tensor& context);

    /// Not typically used directly
    Tensor forward(const Tensor& input) override {
        return forward(input, input);
    }

    std::string to_string() const override;

private:
    std::shared_ptr<MultiheadAttention> mha_;
};

}  // namespace pyflame::nn
