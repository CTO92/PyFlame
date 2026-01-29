#include "pyflame/nn/attention.hpp"
#include "pyflame/ir/node.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/core/tensor_impl.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace pyflame::nn {

// ============================================================================
// MultiheadAttention Implementation
// ============================================================================

MultiheadAttention::MultiheadAttention(int64_t embed_dim,
                                       int64_t num_heads,
                                       float dropout,
                                       bool bias,
                                       bool add_bias_kv,
                                       bool add_zero_attn,
                                       int64_t kdim,
                                       int64_t vdim,
                                       bool batch_first)
    : embed_dim_(embed_dim)
    , num_heads_(num_heads)
    , head_dim_(0)  // Initialized below after validation
    , kdim_(kdim == 0 ? embed_dim : kdim)
    , vdim_(vdim == 0 ? embed_dim : vdim)
    , dropout_p_(dropout)
    , bias_(bias)
    , add_bias_kv_(add_bias_kv)
    , add_zero_attn_(add_zero_attn)
    , batch_first_(batch_first)
{
    set_name("MultiheadAttention");

    // Validate BEFORE computing head_dim to avoid division by zero
    if (num_heads <= 0) {
        throw std::invalid_argument("num_heads must be positive");
    }
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }

    // Now safe to compute head_dim
    head_dim_ = embed_dim / num_heads;

    // Create projection layers
    q_proj_ = register_module("q_proj", std::make_shared<Linear>(embed_dim, embed_dim, bias));
    k_proj_ = register_module("k_proj", std::make_shared<Linear>(kdim_, embed_dim, bias));
    v_proj_ = register_module("v_proj", std::make_shared<Linear>(vdim_, embed_dim, bias));
    out_proj_ = register_module("out_proj", std::make_shared<Linear>(embed_dim, embed_dim, bias));

    if (dropout > 0.0f) {
        dropout_ = register_module("dropout", std::make_shared<Dropout>(dropout));
    }

    if (add_bias_kv_) {
        bias_k_ = Tensor::zeros({1, 1, embed_dim});
        bias_v_ = Tensor::zeros({1, 1, embed_dim});
        register_parameter("bias_k", bias_k_);
        register_parameter("bias_v", bias_v_);
    }
}

std::tuple<Tensor, Tensor> MultiheadAttention::scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask
) {
    // query: [batch, heads, seq_q, head_dim]
    // key: [batch, heads, seq_k, head_dim]
    // value: [batch, heads, seq_k, head_dim]

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // Compute attention scores: [batch, heads, seq_q, seq_k]
    auto scores = matmul(query, key.transpose(-2, -1)) * scale;

    // Apply attention mask if provided
    if (attn_mask.numel() > 0) {
        // Mask positions where attn_mask is -inf or large negative
        scores = scores + attn_mask;
    }

    // Softmax along key dimension
    auto attn_weights = softmax(scores, -1);

    // Apply dropout
    if (dropout_ && is_training()) {
        attn_weights = dropout_->forward(attn_weights);
    }

    // Apply attention to values: [batch, heads, seq_q, head_dim]
    auto attn_output = matmul(attn_weights, value);

    return {attn_output, attn_weights};
}

std::tuple<Tensor, Tensor> MultiheadAttention::forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask,
    bool need_weights
) {
    // Get dimensions
    auto q_shape = query.shape();
    bool is_batched = q_shape.size() == 3;

    int64_t tgt_len, batch_size;
    if (batch_first_) {
        batch_size = q_shape[0];
        tgt_len = q_shape[1];
    } else {
        tgt_len = q_shape[0];
        batch_size = q_shape[1];
    }

    auto k_shape = key.shape();
    int64_t src_len = batch_first_ ? k_shape[1] : k_shape[0];

    // Project Q, K, V
    auto q = q_proj_->forward(query);  // [..., embed_dim]
    auto k = k_proj_->forward(key);
    auto v = v_proj_->forward(value);

    // Reshape for multi-head attention
    // From [batch, seq, embed] to [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim]
    if (batch_first_) {
        q = q.reshape({batch_size, tgt_len, num_heads_, head_dim_}).transpose(1, 2);
        k = k.reshape({batch_size, src_len, num_heads_, head_dim_}).transpose(1, 2);
        v = v.reshape({batch_size, src_len, num_heads_, head_dim_}).transpose(1, 2);
    } else {
        // [seq, batch, embed] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q = q.transpose(0, 1).reshape({batch_size, tgt_len, num_heads_, head_dim_}).transpose(1, 2);
        k = k.transpose(0, 1).reshape({batch_size, src_len, num_heads_, head_dim_}).transpose(1, 2);
        v = v.transpose(0, 1).reshape({batch_size, src_len, num_heads_, head_dim_}).transpose(1, 2);
    }

    // Compute attention
    auto [attn_output, attn_weights] = scaled_dot_product_attention(q, k, v, attn_mask);

    // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed]
    attn_output = attn_output.transpose(1, 2).reshape({batch_size, tgt_len, embed_dim_});

    if (!batch_first_) {
        // Convert back to [seq, batch, embed]
        attn_output = attn_output.transpose(0, 1);
    }

    // Output projection
    attn_output = out_proj_->forward(attn_output);

    if (need_weights) {
        // Average attention weights across heads for visualization
        attn_weights = attn_weights.mean(1);  // [batch, seq_q, seq_k]
        return {attn_output, attn_weights};
    } else {
        return {attn_output, Tensor()};
    }
}

Tensor MultiheadAttention::forward(const Tensor& input) {
    // Self-attention: query = key = value = input
    auto [output, _] = forward(input, input, input, Tensor(), false);
    return output;
}

std::string MultiheadAttention::to_string() const {
    std::ostringstream oss;
    oss << "MultiheadAttention("
        << "embed_dim=" << embed_dim_
        << ", num_heads=" << num_heads_;
    if (dropout_p_ > 0.0f) {
        oss << ", dropout=" << dropout_p_;
    }
    if (batch_first_) {
        oss << ", batch_first=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// SelfAttention Implementation
// ============================================================================

SelfAttention::SelfAttention(int64_t embed_dim,
                             int64_t num_heads,
                             float dropout,
                             bool bias)
{
    set_name("SelfAttention");
    mha_ = register_module("mha",
        std::make_shared<MultiheadAttention>(embed_dim, num_heads, dropout, bias));
}

Tensor SelfAttention::forward(const Tensor& input) {
    auto [output, _] = mha_->forward(input, input, input, Tensor(), false);
    return output;
}

std::string SelfAttention::to_string() const {
    std::ostringstream oss;
    oss << "SelfAttention(embed_dim=" << mha_->embed_dim()
        << ", num_heads=" << mha_->num_heads()
        << ")";
    return oss.str();
}

// ============================================================================
// CrossAttention Implementation
// ============================================================================

CrossAttention::CrossAttention(int64_t embed_dim,
                               int64_t num_heads,
                               float dropout,
                               bool bias)
{
    set_name("CrossAttention");
    mha_ = register_module("mha",
        std::make_shared<MultiheadAttention>(embed_dim, num_heads, dropout, bias));
}

Tensor CrossAttention::forward(const Tensor& query, const Tensor& context) {
    auto [output, _] = mha_->forward(query, context, context, Tensor(), false);
    return output;
}

std::string CrossAttention::to_string() const {
    std::ostringstream oss;
    oss << "CrossAttention(embed_dim=" << mha_->embed_dim()
        << ", num_heads=" << mha_->num_heads()
        << ")";
    return oss.str();
}

}  // namespace pyflame::nn
