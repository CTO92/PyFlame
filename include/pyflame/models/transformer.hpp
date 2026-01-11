#pragma once

#include "pyflame/nn/module.hpp"
#include "pyflame/nn/layers.hpp"
#include "pyflame/core/tensor.hpp"

#include <vector>
#include <memory>
#include <optional>
#include <string>
#include <cmath>

namespace pyflame::models {

// ============================================================================
// Attention Mechanisms
// ============================================================================

/// Scaled dot-product attention
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& mask = std::nullopt,
    float dropout_p = 0.0f,
    bool is_causal = false
);

/// Multi-Head Attention layer
/// Reference: "Attention Is All You Need" (Vaswani et al., 2017)
class MultiHeadAttention : public nn::Module {
public:
    MultiHeadAttention(
        int64_t embed_dim,
        int64_t num_heads,
        float dropout = 0.0f,
        bool bias = true,
        bool add_bias_kv = false,
        bool add_zero_attn = false,
        int64_t kdim = -1,  // -1 means same as embed_dim
        int64_t vdim = -1,
        bool batch_first = true
    );

    /// Forward pass with optional mask and key/value
    /// @param query Query tensor [batch, seq_len, embed_dim] or [seq_len, batch, embed_dim]
    /// @param key Key tensor (optional, defaults to query)
    /// @param value Value tensor (optional, defaults to key)
    /// @param key_padding_mask Mask for padded positions [batch, seq_len]
    /// @param attn_mask Additive attention mask [seq_len, seq_len] or [batch, seq_len, seq_len]
    /// @param need_weights If true, return attention weights
    /// @param is_causal If true, apply causal mask
    /// @return Output tensor and optionally attention weights
    std::pair<Tensor, std::optional<Tensor>> forward(
        const Tensor& query,
        const std::optional<Tensor>& key = std::nullopt,
        const std::optional<Tensor>& value = std::nullopt,
        const std::optional<Tensor>& key_padding_mask = std::nullopt,
        const std::optional<Tensor>& attn_mask = std::nullopt,
        bool need_weights = false,
        bool is_causal = false
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "MultiHeadAttention"; }

private:
    int64_t embed_dim_;
    int64_t num_heads_;
    int64_t head_dim_;
    float dropout_;
    bool batch_first_;

    nn::Linear q_proj_;
    nn::Linear k_proj_;
    nn::Linear v_proj_;
    nn::Linear out_proj_;
    nn::Dropout dropout_layer_;
};

// ============================================================================
// Transformer Layers
// ============================================================================

/// Transformer Encoder Layer
/// Self-attention followed by feed-forward network
class TransformerEncoderLayer : public nn::Module {
public:
    TransformerEncoderLayer(
        int64_t d_model,
        int64_t nhead,
        int64_t dim_feedforward = 2048,
        float dropout = 0.1f,
        const std::string& activation = "relu",
        float layer_norm_eps = 1e-5f,
        bool batch_first = true,
        bool norm_first = false  // Pre-LN vs Post-LN
    );

    /// Forward pass
    /// @param src Source tensor [batch, seq_len, d_model]
    /// @param src_mask Attention mask [seq_len, seq_len]
    /// @param src_key_padding_mask Padding mask [batch, seq_len]
    Tensor forward(
        const Tensor& src,
        const std::optional<Tensor>& src_mask = std::nullopt,
        const std::optional<Tensor>& src_key_padding_mask = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "TransformerEncoderLayer"; }

private:
    MultiHeadAttention self_attn_;
    nn::Linear linear1_;
    nn::Linear linear2_;
    nn::LayerNorm norm1_;
    nn::LayerNorm norm2_;
    nn::Dropout dropout_;
    nn::Dropout dropout1_;
    nn::Dropout dropout2_;
    std::string activation_;
    bool norm_first_;
};

/// Transformer Decoder Layer
/// Self-attention + cross-attention + feed-forward
class TransformerDecoderLayer : public nn::Module {
public:
    TransformerDecoderLayer(
        int64_t d_model,
        int64_t nhead,
        int64_t dim_feedforward = 2048,
        float dropout = 0.1f,
        const std::string& activation = "relu",
        float layer_norm_eps = 1e-5f,
        bool batch_first = true,
        bool norm_first = false
    );

    /// Forward pass
    /// @param tgt Target tensor [batch, tgt_len, d_model]
    /// @param memory Encoder output [batch, src_len, d_model]
    /// @param tgt_mask Target attention mask
    /// @param memory_mask Memory attention mask
    /// @param tgt_key_padding_mask Target padding mask
    /// @param memory_key_padding_mask Memory padding mask
    Tensor forward(
        const Tensor& tgt,
        const Tensor& memory,
        const std::optional<Tensor>& tgt_mask = std::nullopt,
        const std::optional<Tensor>& memory_mask = std::nullopt,
        const std::optional<Tensor>& tgt_key_padding_mask = std::nullopt,
        const std::optional<Tensor>& memory_key_padding_mask = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "TransformerDecoderLayer"; }

private:
    MultiHeadAttention self_attn_;
    MultiHeadAttention multihead_attn_;
    nn::Linear linear1_;
    nn::Linear linear2_;
    nn::LayerNorm norm1_;
    nn::LayerNorm norm2_;
    nn::LayerNorm norm3_;
    nn::Dropout dropout_;
    nn::Dropout dropout1_;
    nn::Dropout dropout2_;
    nn::Dropout dropout3_;
    std::string activation_;
    bool norm_first_;
};

/// Transformer Encoder (stack of encoder layers)
class TransformerEncoder : public nn::Module {
public:
    TransformerEncoder(
        const TransformerEncoderLayer& encoder_layer,
        int num_layers,
        const std::optional<nn::LayerNorm>& norm = std::nullopt
    );

    Tensor forward(
        const Tensor& src,
        const std::optional<Tensor>& mask = std::nullopt,
        const std::optional<Tensor>& src_key_padding_mask = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "TransformerEncoder"; }

private:
    std::vector<std::shared_ptr<TransformerEncoderLayer>> layers_;
    std::optional<nn::LayerNorm> norm_;
};

/// Transformer Decoder (stack of decoder layers)
class TransformerDecoder : public nn::Module {
public:
    TransformerDecoder(
        const TransformerDecoderLayer& decoder_layer,
        int num_layers,
        const std::optional<nn::LayerNorm>& norm = std::nullopt
    );

    Tensor forward(
        const Tensor& tgt,
        const Tensor& memory,
        const std::optional<Tensor>& tgt_mask = std::nullopt,
        const std::optional<Tensor>& memory_mask = std::nullopt,
        const std::optional<Tensor>& tgt_key_padding_mask = std::nullopt,
        const std::optional<Tensor>& memory_key_padding_mask = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "TransformerDecoder"; }

private:
    std::vector<std::shared_ptr<TransformerDecoderLayer>> layers_;
    std::optional<nn::LayerNorm> norm_;
};

// ============================================================================
// Full Transformer Model
// ============================================================================

/// Configuration for Transformer model
struct TransformerConfig {
    int64_t d_model = 512;
    int64_t nhead = 8;
    int64_t num_encoder_layers = 6;
    int64_t num_decoder_layers = 6;
    int64_t dim_feedforward = 2048;
    float dropout = 0.1f;
    std::string activation = "relu";
    bool batch_first = true;
    bool norm_first = false;
    float layer_norm_eps = 1e-5f;

    // Common presets
    static TransformerConfig Base();
    static TransformerConfig Large();
};

/// Full Transformer model with encoder and decoder
class Transformer : public nn::Module {
public:
    explicit Transformer(const TransformerConfig& config = TransformerConfig());

    /// Forward pass
    /// @param src Source sequence [batch, src_len, d_model]
    /// @param tgt Target sequence [batch, tgt_len, d_model]
    /// @param src_mask Source attention mask
    /// @param tgt_mask Target attention mask
    /// @param memory_mask Memory attention mask
    /// @param src_key_padding_mask Source padding mask
    /// @param tgt_key_padding_mask Target padding mask
    /// @param memory_key_padding_mask Memory padding mask
    Tensor forward(
        const Tensor& src,
        const Tensor& tgt,
        const std::optional<Tensor>& src_mask = std::nullopt,
        const std::optional<Tensor>& tgt_mask = std::nullopt,
        const std::optional<Tensor>& memory_mask = std::nullopt,
        const std::optional<Tensor>& src_key_padding_mask = std::nullopt,
        const std::optional<Tensor>& tgt_key_padding_mask = std::nullopt,
        const std::optional<Tensor>& memory_key_padding_mask = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    /// Generate causal mask for autoregressive decoding
    static Tensor generate_square_subsequent_mask(int64_t sz);

    std::string name() const override { return "Transformer"; }

private:
    TransformerConfig config_;
    TransformerEncoder encoder_;
    TransformerDecoder decoder_;
};

// ============================================================================
// Positional Encoding
// ============================================================================

/// Sinusoidal positional encoding
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
class PositionalEncoding : public nn::Module {
public:
    PositionalEncoding(
        int64_t d_model,
        float dropout = 0.1f,
        int64_t max_len = 5000
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "PositionalEncoding"; }

private:
    nn::Dropout dropout_;
    Tensor pe_;  // [1, max_len, d_model]
};

/// Learned positional embeddings
class LearnedPositionalEmbedding : public nn::Module {
public:
    LearnedPositionalEmbedding(
        int64_t num_positions,
        int64_t embedding_dim
    );

    Tensor forward(const Tensor& x) override;

    /// Get positions for input of given length
    Tensor forward(const Tensor& x, int64_t start_pos);

    std::string name() const override { return "LearnedPositionalEmbedding"; }

private:
    nn::Embedding position_embeddings_;
    int64_t num_positions_;
};

// ============================================================================
// BERT-style Models
// ============================================================================

/// BERT configuration
struct BertConfig {
    int64_t vocab_size = 30522;
    int64_t hidden_size = 768;
    int64_t num_hidden_layers = 12;
    int64_t num_attention_heads = 12;
    int64_t intermediate_size = 3072;
    float hidden_dropout_prob = 0.1f;
    float attention_probs_dropout_prob = 0.1f;
    int64_t max_position_embeddings = 512;
    int64_t type_vocab_size = 2;
    float layer_norm_eps = 1e-12f;
    int64_t pad_token_id = 0;

    // Presets
    static BertConfig Base();
    static BertConfig Large();
};

/// BERT Embeddings: word + position + token_type
class BertEmbeddings : public nn::Module {
public:
    explicit BertEmbeddings(const BertConfig& config);

    /// @param input_ids Token IDs [batch, seq_len]
    /// @param token_type_ids Segment IDs [batch, seq_len]
    /// @param position_ids Position IDs [batch, seq_len]
    Tensor forward(
        const Tensor& input_ids,
        const std::optional<Tensor>& token_type_ids = std::nullopt,
        const std::optional<Tensor>& position_ids = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "BertEmbeddings"; }

private:
    nn::Embedding word_embeddings_;
    nn::Embedding position_embeddings_;
    nn::Embedding token_type_embeddings_;
    nn::LayerNorm layer_norm_;
    nn::Dropout dropout_;
};

/// BERT Model (encoder-only transformer)
class BertModel : public nn::Module {
public:
    explicit BertModel(const BertConfig& config);

    /// Forward pass
    /// @param input_ids Token IDs [batch, seq_len]
    /// @param attention_mask Attention mask [batch, seq_len]
    /// @param token_type_ids Segment IDs [batch, seq_len]
    /// @return Tuple of (last_hidden_state, pooler_output)
    std::pair<Tensor, Tensor> forward(
        const Tensor& input_ids,
        const std::optional<Tensor>& attention_mask = std::nullopt,
        const std::optional<Tensor>& token_type_ids = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    /// Get hidden size
    int64_t hidden_size() const { return config_.hidden_size; }

    std::string name() const override { return "BertModel"; }

private:
    BertConfig config_;
    BertEmbeddings embeddings_;
    TransformerEncoder encoder_;
    nn::Linear pooler_dense_;
};

/// BERT for sequence classification
class BertForSequenceClassification : public nn::Module {
public:
    BertForSequenceClassification(const BertConfig& config, int64_t num_labels);

    /// @param input_ids Token IDs [batch, seq_len]
    /// @param attention_mask Attention mask [batch, seq_len]
    /// @param token_type_ids Segment IDs [batch, seq_len]
    /// @param labels Optional labels for computing loss [batch]
    /// @return Logits [batch, num_labels] (and loss if labels provided)
    std::pair<Tensor, std::optional<Tensor>> forward(
        const Tensor& input_ids,
        const std::optional<Tensor>& attention_mask = std::nullopt,
        const std::optional<Tensor>& token_type_ids = std::nullopt,
        const std::optional<Tensor>& labels = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "BertForSequenceClassification"; }

private:
    BertModel bert_;
    nn::Dropout dropout_;
    nn::Linear classifier_;
    int64_t num_labels_;
};

/// BERT for masked language modeling
class BertForMaskedLM : public nn::Module {
public:
    explicit BertForMaskedLM(const BertConfig& config);

    /// @param input_ids Token IDs [batch, seq_len]
    /// @param attention_mask Attention mask [batch, seq_len]
    /// @param labels Labels for MLM loss [batch, seq_len]
    /// @return Prediction scores [batch, seq_len, vocab_size]
    std::pair<Tensor, std::optional<Tensor>> forward(
        const Tensor& input_ids,
        const std::optional<Tensor>& attention_mask = std::nullopt,
        const std::optional<Tensor>& labels = std::nullopt
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "BertForMaskedLM"; }

private:
    BertModel bert_;
    nn::Linear predictions_dense_;
    nn::LayerNorm predictions_layer_norm_;
    nn::Linear predictions_decoder_;  // Tied with embeddings
    BertConfig config_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// Create BERT base model
inline std::unique_ptr<BertModel> bert_base() {
    return std::make_unique<BertModel>(BertConfig::Base());
}

/// Create BERT large model
inline std::unique_ptr<BertModel> bert_large() {
    return std::make_unique<BertModel>(BertConfig::Large());
}

/// Create base Transformer
inline std::unique_ptr<Transformer> transformer_base() {
    return std::make_unique<Transformer>(TransformerConfig::Base());
}

/// Create large Transformer
inline std::unique_ptr<Transformer> transformer_large() {
    return std::make_unique<Transformer>(TransformerConfig::Large());
}

}  // namespace pyflame::models
