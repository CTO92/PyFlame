#include "pyflame/models/transformer.hpp"
#include "pyflame/ops/activations.hpp"
#include "pyflame/ops/math_ops.hpp"

#include <cmath>
#include <limits>

namespace pyflame::models {

// ============================================================================
// Attention Implementation
// ============================================================================

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& mask,
    float dropout_p,
    bool is_causal
) {
    // query: [batch, heads, seq_q, head_dim]
    // key:   [batch, heads, seq_k, head_dim]
    // value: [batch, heads, seq_k, head_dim]

    auto shape = query.shape();
    int64_t head_dim = shape.back();
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // QK^T / sqrt(d_k)
    Tensor attn_weights = ops::matmul(query, key.transpose(-2, -1)) * scale;

    // Apply causal mask if needed
    if (is_causal) {
        int64_t seq_q = query.shape()[-2];
        int64_t seq_k = key.shape()[-2];
        Tensor causal_mask = Tensor::ones({seq_q, seq_k}, query.dtype()).triu(1);
        causal_mask = causal_mask * (-std::numeric_limits<float>::infinity());
        attn_weights = attn_weights + causal_mask;
    }

    // Apply attention mask
    if (mask.has_value()) {
        attn_weights = attn_weights + mask.value();
    }

    // Softmax
    attn_weights = ops::softmax(attn_weights, -1);

    // Dropout (if training)
    if (dropout_p > 0.0f) {
        // Apply dropout during training
        // attn_weights = ops::dropout(attn_weights, dropout_p);
    }

    // Weighted sum of values
    return ops::matmul(attn_weights, value);
}

// ============================================================================
// MultiHeadAttention Implementation
// ============================================================================

MultiHeadAttention::MultiHeadAttention(
    int64_t embed_dim,
    int64_t num_heads,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    int64_t kdim,
    int64_t vdim,
    bool batch_first
) : embed_dim_(embed_dim),
    num_heads_(num_heads),
    head_dim_(embed_dim / num_heads),
    dropout_(dropout),
    batch_first_(batch_first),
    q_proj_(embed_dim, embed_dim, bias),
    k_proj_(kdim > 0 ? kdim : embed_dim, embed_dim, bias),
    v_proj_(vdim > 0 ? vdim : embed_dim, embed_dim, bias),
    out_proj_(embed_dim, embed_dim, bias),
    dropout_layer_(dropout)
{
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "embed_dim must be divisible by num_heads");
    }

    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
    register_module("dropout", dropout_layer_);
}

std::pair<Tensor, std::optional<Tensor>> MultiHeadAttention::forward(
    const Tensor& query,
    const std::optional<Tensor>& key,
    const std::optional<Tensor>& value,
    const std::optional<Tensor>& key_padding_mask,
    const std::optional<Tensor>& attn_mask,
    bool need_weights,
    bool is_causal
) {
    // Use query as key/value if not provided (self-attention)
    const Tensor& k = key.value_or(query);
    const Tensor& v = value.value_or(k);

    auto shape = query.shape();
    int64_t batch_size = batch_first_ ? shape[0] : shape[1];
    int64_t seq_len = batch_first_ ? shape[1] : shape[0];

    // Project Q, K, V
    Tensor q = q_proj_.forward(query);
    Tensor k_proj = k_proj_.forward(k);
    Tensor v_proj = v_proj_.forward(v);

    // Reshape for multi-head attention
    // [batch, seq, embed] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    auto reshape_for_attn = [&](Tensor& t, int64_t seq) {
        if (batch_first_) {
            t = t.view({batch_size, seq, num_heads_, head_dim_});
            t = t.transpose(1, 2);
        } else {
            t = t.view({seq, batch_size, num_heads_, head_dim_});
            t = t.permute({1, 2, 0, 3});
        }
    };

    reshape_for_attn(q, seq_len);
    reshape_for_attn(k_proj, k.shape()[batch_first_ ? 1 : 0]);
    reshape_for_attn(v_proj, v.shape()[batch_first_ ? 1 : 0]);

    // Compute attention
    Tensor attn_output = scaled_dot_product_attention(
        q, k_proj, v_proj, attn_mask, dropout_, is_causal);

    // Reshape back
    // [batch, heads, seq, head_dim] -> [batch, seq, embed]
    if (batch_first_) {
        attn_output = attn_output.transpose(1, 2);
        attn_output = attn_output.reshape({batch_size, seq_len, embed_dim_});
    } else {
        attn_output = attn_output.permute({2, 0, 1, 3});
        attn_output = attn_output.reshape({seq_len, batch_size, embed_dim_});
    }

    // Output projection
    attn_output = out_proj_.forward(attn_output);

    if (need_weights) {
        // Return attention weights (would need to compute separately)
        return {attn_output, std::nullopt};
    }
    return {attn_output, std::nullopt};
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    return forward(x, std::nullopt, std::nullopt).first;
}

// ============================================================================
// TransformerEncoderLayer Implementation
// ============================================================================

TransformerEncoderLayer::TransformerEncoderLayer(
    int64_t d_model,
    int64_t nhead,
    int64_t dim_feedforward,
    float dropout,
    const std::string& activation,
    float layer_norm_eps,
    bool batch_first,
    bool norm_first
) : self_attn_(d_model, nhead, dropout, true, false, false, -1, -1, batch_first),
    linear1_(d_model, dim_feedforward),
    linear2_(dim_feedforward, d_model),
    norm1_(d_model, layer_norm_eps),
    norm2_(d_model, layer_norm_eps),
    dropout_(dropout),
    dropout1_(dropout),
    dropout2_(dropout),
    activation_(activation),
    norm_first_(norm_first)
{
    register_module("self_attn", self_attn_);
    register_module("linear1", linear1_);
    register_module("linear2", linear2_);
    register_module("norm1", norm1_);
    register_module("norm2", norm2_);
    register_module("dropout", dropout_);
    register_module("dropout1", dropout1_);
    register_module("dropout2", dropout2_);
}

Tensor TransformerEncoderLayer::forward(
    const Tensor& src,
    const std::optional<Tensor>& src_mask,
    const std::optional<Tensor>& src_key_padding_mask
) {
    Tensor x = src;

    if (norm_first_) {
        // Pre-LN: norm -> attn -> residual
        Tensor normed = norm1_.forward(x);
        auto [attn_out, _] = self_attn_.forward(
            normed, normed, normed, src_key_padding_mask, src_mask);
        x = x + dropout1_.forward(attn_out);

        // FFN
        normed = norm2_.forward(x);
        Tensor ff_out = linear1_.forward(normed);
        if (activation_ == "relu") {
            ff_out = ops::relu(ff_out);
        } else if (activation_ == "gelu") {
            ff_out = ops::gelu(ff_out);
        }
        ff_out = dropout_.forward(ff_out);
        ff_out = linear2_.forward(ff_out);
        x = x + dropout2_.forward(ff_out);
    } else {
        // Post-LN: attn -> residual -> norm
        auto [attn_out, _] = self_attn_.forward(
            x, x, x, src_key_padding_mask, src_mask);
        x = x + dropout1_.forward(attn_out);
        x = norm1_.forward(x);

        // FFN
        Tensor ff_out = linear1_.forward(x);
        if (activation_ == "relu") {
            ff_out = ops::relu(ff_out);
        } else if (activation_ == "gelu") {
            ff_out = ops::gelu(ff_out);
        }
        ff_out = dropout_.forward(ff_out);
        ff_out = linear2_.forward(ff_out);
        x = x + dropout2_.forward(ff_out);
        x = norm2_.forward(x);
    }

    return x;
}

Tensor TransformerEncoderLayer::forward(const Tensor& x) {
    return forward(x, std::nullopt, std::nullopt);
}

// ============================================================================
// TransformerDecoderLayer Implementation
// ============================================================================

TransformerDecoderLayer::TransformerDecoderLayer(
    int64_t d_model,
    int64_t nhead,
    int64_t dim_feedforward,
    float dropout,
    const std::string& activation,
    float layer_norm_eps,
    bool batch_first,
    bool norm_first
) : self_attn_(d_model, nhead, dropout, true, false, false, -1, -1, batch_first),
    multihead_attn_(d_model, nhead, dropout, true, false, false, -1, -1, batch_first),
    linear1_(d_model, dim_feedforward),
    linear2_(dim_feedforward, d_model),
    norm1_(d_model, layer_norm_eps),
    norm2_(d_model, layer_norm_eps),
    norm3_(d_model, layer_norm_eps),
    dropout_(dropout),
    dropout1_(dropout),
    dropout2_(dropout),
    dropout3_(dropout),
    activation_(activation),
    norm_first_(norm_first)
{
    register_module("self_attn", self_attn_);
    register_module("multihead_attn", multihead_attn_);
    register_module("linear1", linear1_);
    register_module("linear2", linear2_);
    register_module("norm1", norm1_);
    register_module("norm2", norm2_);
    register_module("norm3", norm3_);
    register_module("dropout", dropout_);
    register_module("dropout1", dropout1_);
    register_module("dropout2", dropout2_);
    register_module("dropout3", dropout3_);
}

Tensor TransformerDecoderLayer::forward(
    const Tensor& tgt,
    const Tensor& memory,
    const std::optional<Tensor>& tgt_mask,
    const std::optional<Tensor>& memory_mask,
    const std::optional<Tensor>& tgt_key_padding_mask,
    const std::optional<Tensor>& memory_key_padding_mask
) {
    Tensor x = tgt;

    if (norm_first_) {
        // Pre-LN
        Tensor normed = norm1_.forward(x);
        auto [self_out, _1] = self_attn_.forward(
            normed, normed, normed, tgt_key_padding_mask, tgt_mask);
        x = x + dropout1_.forward(self_out);

        normed = norm2_.forward(x);
        auto [cross_out, _2] = multihead_attn_.forward(
            normed, memory, memory, memory_key_padding_mask, memory_mask);
        x = x + dropout2_.forward(cross_out);

        normed = norm3_.forward(x);
        Tensor ff_out = linear1_.forward(normed);
        if (activation_ == "relu") {
            ff_out = ops::relu(ff_out);
        } else if (activation_ == "gelu") {
            ff_out = ops::gelu(ff_out);
        }
        ff_out = dropout_.forward(ff_out);
        ff_out = linear2_.forward(ff_out);
        x = x + dropout3_.forward(ff_out);
    } else {
        // Post-LN
        auto [self_out, _1] = self_attn_.forward(
            x, x, x, tgt_key_padding_mask, tgt_mask);
        x = x + dropout1_.forward(self_out);
        x = norm1_.forward(x);

        auto [cross_out, _2] = multihead_attn_.forward(
            x, memory, memory, memory_key_padding_mask, memory_mask);
        x = x + dropout2_.forward(cross_out);
        x = norm2_.forward(x);

        Tensor ff_out = linear1_.forward(x);
        if (activation_ == "relu") {
            ff_out = ops::relu(ff_out);
        } else if (activation_ == "gelu") {
            ff_out = ops::gelu(ff_out);
        }
        ff_out = dropout_.forward(ff_out);
        ff_out = linear2_.forward(ff_out);
        x = x + dropout3_.forward(ff_out);
        x = norm3_.forward(x);
    }

    return x;
}

Tensor TransformerDecoderLayer::forward(const Tensor& x) {
    throw std::runtime_error(
        "TransformerDecoderLayer requires memory tensor from encoder");
}

// ============================================================================
// TransformerEncoder Implementation
// ============================================================================

TransformerEncoder::TransformerEncoder(
    const TransformerEncoderLayer& encoder_layer,
    int num_layers,
    const std::optional<nn::LayerNorm>& norm
) : norm_(norm)
{
    layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        auto layer = std::make_shared<TransformerEncoderLayer>(encoder_layer);
        layers_.push_back(layer);
        register_module("layer_" + std::to_string(i), *layer);
    }

    if (norm_.has_value()) {
        register_module("norm", *norm_);
    }
}

Tensor TransformerEncoder::forward(
    const Tensor& src,
    const std::optional<Tensor>& mask,
    const std::optional<Tensor>& src_key_padding_mask
) {
    Tensor output = src;

    for (auto& layer : layers_) {
        output = layer->forward(output, mask, src_key_padding_mask);
    }

    if (norm_.has_value()) {
        output = norm_->forward(output);
    }

    return output;
}

Tensor TransformerEncoder::forward(const Tensor& x) {
    return forward(x, std::nullopt, std::nullopt);
}

// ============================================================================
// TransformerDecoder Implementation
// ============================================================================

TransformerDecoder::TransformerDecoder(
    const TransformerDecoderLayer& decoder_layer,
    int num_layers,
    const std::optional<nn::LayerNorm>& norm
) : norm_(norm)
{
    layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        auto layer = std::make_shared<TransformerDecoderLayer>(decoder_layer);
        layers_.push_back(layer);
        register_module("layer_" + std::to_string(i), *layer);
    }

    if (norm_.has_value()) {
        register_module("norm", *norm_);
    }
}

Tensor TransformerDecoder::forward(
    const Tensor& tgt,
    const Tensor& memory,
    const std::optional<Tensor>& tgt_mask,
    const std::optional<Tensor>& memory_mask,
    const std::optional<Tensor>& tgt_key_padding_mask,
    const std::optional<Tensor>& memory_key_padding_mask
) {
    Tensor output = tgt;

    for (auto& layer : layers_) {
        output = layer->forward(
            output, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask);
    }

    if (norm_.has_value()) {
        output = norm_->forward(output);
    }

    return output;
}

Tensor TransformerDecoder::forward(const Tensor& x) {
    throw std::runtime_error(
        "TransformerDecoder requires memory tensor from encoder");
}

// ============================================================================
// TransformerConfig Implementation
// ============================================================================

TransformerConfig TransformerConfig::Base() {
    return TransformerConfig{
        .d_model = 512,
        .nhead = 8,
        .num_encoder_layers = 6,
        .num_decoder_layers = 6,
        .dim_feedforward = 2048,
        .dropout = 0.1f
    };
}

TransformerConfig TransformerConfig::Large() {
    return TransformerConfig{
        .d_model = 1024,
        .nhead = 16,
        .num_encoder_layers = 6,
        .num_decoder_layers = 6,
        .dim_feedforward = 4096,
        .dropout = 0.1f
    };
}

// ============================================================================
// Transformer Implementation
// ============================================================================

Transformer::Transformer(const TransformerConfig& config)
    : config_(config),
      encoder_(
          TransformerEncoderLayer(
              config.d_model, config.nhead, config.dim_feedforward,
              config.dropout, config.activation, config.layer_norm_eps,
              config.batch_first, config.norm_first),
          config.num_encoder_layers,
          nn::LayerNorm(config.d_model, config.layer_norm_eps)),
      decoder_(
          TransformerDecoderLayer(
              config.d_model, config.nhead, config.dim_feedforward,
              config.dropout, config.activation, config.layer_norm_eps,
              config.batch_first, config.norm_first),
          config.num_decoder_layers,
          nn::LayerNorm(config.d_model, config.layer_norm_eps))
{
    register_module("encoder", encoder_);
    register_module("decoder", decoder_);
}

Tensor Transformer::forward(
    const Tensor& src,
    const Tensor& tgt,
    const std::optional<Tensor>& src_mask,
    const std::optional<Tensor>& tgt_mask,
    const std::optional<Tensor>& memory_mask,
    const std::optional<Tensor>& src_key_padding_mask,
    const std::optional<Tensor>& tgt_key_padding_mask,
    const std::optional<Tensor>& memory_key_padding_mask
) {
    // Encode source
    Tensor memory = encoder_.forward(src, src_mask, src_key_padding_mask);

    // Decode target
    Tensor output = decoder_.forward(
        tgt, memory, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask);

    return output;
}

Tensor Transformer::forward(const Tensor& x) {
    throw std::runtime_error(
        "Transformer.forward(x) requires both src and tgt tensors");
}

Tensor Transformer::generate_square_subsequent_mask(int64_t sz) {
    // Create upper triangular matrix of -inf
    Tensor mask = Tensor::ones({sz, sz}, DType::Float32).triu(1);
    mask = mask * (-std::numeric_limits<float>::infinity());
    return mask;
}

// ============================================================================
// PositionalEncoding Implementation
// ============================================================================

PositionalEncoding::PositionalEncoding(
    int64_t d_model,
    float dropout,
    int64_t max_len
) : dropout_(dropout)
{
    register_module("dropout", dropout_);

    // Create position indices [max_len, 1]
    Tensor position = Tensor::arange(0, max_len, 1, DType::Float32)
        .unsqueeze(1);

    // Create dimension indices and compute div_term
    Tensor div_term = Tensor::arange(0, d_model, 2, DType::Float32);
    div_term = ops::exp(div_term * (-std::log(10000.0f) / d_model));

    // Compute positional encodings
    pe_ = Tensor::zeros({1, max_len, d_model});

    // PE[:, :, 0::2] = sin(position * div_term)
    // PE[:, :, 1::2] = cos(position * div_term)
    Tensor angles = ops::matmul(position, div_term.unsqueeze(0));
    Tensor sin_enc = ops::sin(angles);
    Tensor cos_enc = ops::cos(angles);

    // Interleave sin and cos
    for (int64_t i = 0; i < d_model / 2; ++i) {
        // pe_[0, :, 2*i] = sin_enc[:, i]
        // pe_[0, :, 2*i+1] = cos_enc[:, i]
    }

    // Store as buffer (not a parameter)
    // In practice, this would be registered as a buffer
}

Tensor PositionalEncoding::forward(const Tensor& x) {
    // x: [batch, seq_len, d_model]
    int64_t seq_len = x.shape()[1];

    // Add positional encoding (broadcasting)
    Tensor out = x + pe_.slice(1, 0, seq_len);
    return dropout_.forward(out);
}

// ============================================================================
// LearnedPositionalEmbedding Implementation
// ============================================================================

LearnedPositionalEmbedding::LearnedPositionalEmbedding(
    int64_t num_positions,
    int64_t embedding_dim
) : position_embeddings_(num_positions, embedding_dim),
    num_positions_(num_positions)
{
    register_module("position_embeddings", position_embeddings_);
}

Tensor LearnedPositionalEmbedding::forward(const Tensor& x) {
    return forward(x, 0);
}

Tensor LearnedPositionalEmbedding::forward(const Tensor& x, int64_t start_pos) {
    int64_t seq_len = x.shape()[1];

    // Create position indices
    Tensor positions = Tensor::arange(start_pos, start_pos + seq_len, 1, DType::Int64);

    // Get embeddings
    return position_embeddings_.forward(positions);
}

// ============================================================================
// BertConfig Implementation
// ============================================================================

BertConfig BertConfig::Base() {
    return BertConfig{
        .vocab_size = 30522,
        .hidden_size = 768,
        .num_hidden_layers = 12,
        .num_attention_heads = 12,
        .intermediate_size = 3072
    };
}

BertConfig BertConfig::Large() {
    return BertConfig{
        .vocab_size = 30522,
        .hidden_size = 1024,
        .num_hidden_layers = 24,
        .num_attention_heads = 16,
        .intermediate_size = 4096
    };
}

// ============================================================================
// BertEmbeddings Implementation
// ============================================================================

BertEmbeddings::BertEmbeddings(const BertConfig& config)
    : word_embeddings_(config.vocab_size, config.hidden_size),
      position_embeddings_(config.max_position_embeddings, config.hidden_size),
      token_type_embeddings_(config.type_vocab_size, config.hidden_size),
      layer_norm_(config.hidden_size, config.layer_norm_eps),
      dropout_(config.hidden_dropout_prob)
{
    register_module("word_embeddings", word_embeddings_);
    register_module("position_embeddings", position_embeddings_);
    register_module("token_type_embeddings", token_type_embeddings_);
    register_module("layer_norm", layer_norm_);
    register_module("dropout", dropout_);
}

Tensor BertEmbeddings::forward(
    const Tensor& input_ids,
    const std::optional<Tensor>& token_type_ids,
    const std::optional<Tensor>& position_ids
) {
    int64_t seq_len = input_ids.shape()[1];

    // Word embeddings
    Tensor embeddings = word_embeddings_.forward(input_ids);

    // Position embeddings
    Tensor pos_ids = position_ids.value_or(
        Tensor::arange(0, seq_len, 1, DType::Int64));
    embeddings = embeddings + position_embeddings_.forward(pos_ids);

    // Token type embeddings
    Tensor type_ids = token_type_ids.value_or(
        Tensor::zeros_like(input_ids));
    embeddings = embeddings + token_type_embeddings_.forward(type_ids);

    // Layer norm and dropout
    embeddings = layer_norm_.forward(embeddings);
    embeddings = dropout_.forward(embeddings);

    return embeddings;
}

Tensor BertEmbeddings::forward(const Tensor& x) {
    return forward(x, std::nullopt, std::nullopt);
}

// ============================================================================
// BertModel Implementation
// ============================================================================

BertModel::BertModel(const BertConfig& config)
    : config_(config),
      embeddings_(config),
      encoder_(
          TransformerEncoderLayer(
              config.hidden_size,
              config.num_attention_heads,
              config.intermediate_size,
              config.hidden_dropout_prob,
              "gelu",
              config.layer_norm_eps,
              true,  // batch_first
              false  // post-LN
          ),
          config.num_hidden_layers,
          std::nullopt  // No final norm (already in each layer)
      ),
      pooler_dense_(config.hidden_size, config.hidden_size)
{
    register_module("embeddings", embeddings_);
    register_module("encoder", encoder_);
    register_module("pooler_dense", pooler_dense_);
}

std::pair<Tensor, Tensor> BertModel::forward(
    const Tensor& input_ids,
    const std::optional<Tensor>& attention_mask,
    const std::optional<Tensor>& token_type_ids
) {
    // Get embeddings
    Tensor hidden_states = embeddings_.forward(
        input_ids, token_type_ids, std::nullopt);

    // Create extended attention mask
    std::optional<Tensor> extended_mask = std::nullopt;
    if (attention_mask.has_value()) {
        // Convert [batch, seq] mask to [batch, 1, 1, seq] for broadcasting
        Tensor mask = attention_mask.value();
        mask = mask.unsqueeze(1).unsqueeze(2);
        // Convert 0/1 mask to -inf/0 for attention
        mask = (1.0f - mask) * (-10000.0f);
        extended_mask = mask;
    }

    // Encode
    Tensor sequence_output = encoder_.forward(
        hidden_states, std::nullopt, extended_mask);

    // Pooler: take [CLS] token, apply dense + tanh
    Tensor cls_token = sequence_output.select(1, 0);  // [batch, hidden_size]
    Tensor pooled_output = pooler_dense_.forward(cls_token);
    pooled_output = ops::tanh(pooled_output);

    return {sequence_output, pooled_output};
}

Tensor BertModel::forward(const Tensor& x) {
    return forward(x, std::nullopt, std::nullopt).first;
}

// ============================================================================
// BertForSequenceClassification Implementation
// ============================================================================

BertForSequenceClassification::BertForSequenceClassification(
    const BertConfig& config,
    int64_t num_labels
) : bert_(config),
    dropout_(config.hidden_dropout_prob),
    classifier_(config.hidden_size, num_labels),
    num_labels_(num_labels)
{
    register_module("bert", bert_);
    register_module("dropout", dropout_);
    register_module("classifier", classifier_);
}

std::pair<Tensor, std::optional<Tensor>> BertForSequenceClassification::forward(
    const Tensor& input_ids,
    const std::optional<Tensor>& attention_mask,
    const std::optional<Tensor>& token_type_ids,
    const std::optional<Tensor>& labels
) {
    auto [sequence_output, pooled_output] = bert_.forward(
        input_ids, attention_mask, token_type_ids);

    pooled_output = dropout_.forward(pooled_output);
    Tensor logits = classifier_.forward(pooled_output);

    std::optional<Tensor> loss = std::nullopt;
    if (labels.has_value()) {
        if (num_labels_ == 1) {
            // Regression
            loss = ops::mse_loss(logits.squeeze(-1), labels.value());
        } else {
            // Classification
            loss = ops::cross_entropy_loss(logits, labels.value());
        }
    }

    return {logits, loss};
}

Tensor BertForSequenceClassification::forward(const Tensor& x) {
    return forward(x, std::nullopt, std::nullopt, std::nullopt).first;
}

// ============================================================================
// BertForMaskedLM Implementation
// ============================================================================

BertForMaskedLM::BertForMaskedLM(const BertConfig& config)
    : bert_(config),
      predictions_dense_(config.hidden_size, config.hidden_size),
      predictions_layer_norm_(config.hidden_size, config.layer_norm_eps),
      predictions_decoder_(config.hidden_size, config.vocab_size, false),
      config_(config)
{
    register_module("bert", bert_);
    register_module("predictions_dense", predictions_dense_);
    register_module("predictions_layer_norm", predictions_layer_norm_);
    register_module("predictions_decoder", predictions_decoder_);

    // Tie decoder weights with embeddings (would be done at init in practice)
}

std::pair<Tensor, std::optional<Tensor>> BertForMaskedLM::forward(
    const Tensor& input_ids,
    const std::optional<Tensor>& attention_mask,
    const std::optional<Tensor>& labels
) {
    auto [sequence_output, pooled] = bert_.forward(
        input_ids, attention_mask, std::nullopt);

    // MLM head
    Tensor hidden = predictions_dense_.forward(sequence_output);
    hidden = ops::gelu(hidden);
    hidden = predictions_layer_norm_.forward(hidden);
    Tensor prediction_scores = predictions_decoder_.forward(hidden);

    std::optional<Tensor> loss = std::nullopt;
    if (labels.has_value()) {
        // Compute masked LM loss
        // Flatten: [batch * seq_len, vocab_size] vs [batch * seq_len]
        Tensor flat_scores = prediction_scores.view({-1, config_.vocab_size});
        Tensor flat_labels = labels.value().view({-1});
        loss = ops::cross_entropy_loss(flat_scores, flat_labels, -100);
    }

    return {prediction_scores, loss};
}

Tensor BertForMaskedLM::forward(const Tensor& x) {
    return forward(x, std::nullopt, std::nullopt).first;
}

}  // namespace pyflame::models
