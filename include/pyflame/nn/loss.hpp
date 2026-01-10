#pragma once

#include "pyflame/core/tensor.hpp"

#include <string>

namespace pyflame::nn {

// Reduction mode for loss functions
enum class Reduction {
    NONE,   // No reduction - return per-element loss
    MEAN,   // Mean of all elements
    SUM     // Sum of all elements
};

// ============================================================================
// MSELoss - Mean Squared Error Loss
// ============================================================================
// L = (1/n) * sum((pred - target)^2)  [when reduction=MEAN]

class MSELoss {
public:
    explicit MSELoss(Reduction reduction = Reduction::MEAN);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
};

// ============================================================================
// L1Loss - Mean Absolute Error Loss
// ============================================================================
// L = (1/n) * sum(|pred - target|)  [when reduction=MEAN]

class L1Loss {
public:
    explicit L1Loss(Reduction reduction = Reduction::MEAN);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
};

// ============================================================================
// SmoothL1Loss (Huber Loss)
// ============================================================================
// L = 0.5 * x^2 / beta,     if |x| < beta
//   = |x| - 0.5 * beta,     otherwise
// where x = pred - target

class SmoothL1Loss {
public:
    explicit SmoothL1Loss(Reduction reduction = Reduction::MEAN, float beta = 1.0f);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
    float beta_;
};

// ============================================================================
// HuberLoss - Same as SmoothL1Loss but with configurable delta
// ============================================================================

class HuberLoss {
public:
    explicit HuberLoss(Reduction reduction = Reduction::MEAN, float delta = 1.0f);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
    float delta_;
};

// ============================================================================
// BCELoss - Binary Cross Entropy Loss
// ============================================================================
// L = -[target * log(input) + (1 - target) * log(1 - input)]
// Input should be probabilities in [0, 1] (apply sigmoid first)

class BCELoss {
public:
    explicit BCELoss(Reduction reduction = Reduction::MEAN);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
};

// ============================================================================
// BCEWithLogitsLoss - Binary Cross Entropy with Sigmoid
// ============================================================================
// More numerically stable than BCELoss + Sigmoid
// L = max(x, 0) - x*t + log(1 + exp(-|x|))
// where x = input (logits), t = target

class BCEWithLogitsLoss {
public:
    explicit BCEWithLogitsLoss(Reduction reduction = Reduction::MEAN);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
};

// ============================================================================
// NLLLoss - Negative Log Likelihood Loss
// ============================================================================
// L = -log(input[class])
// Input should be log-probabilities (apply log_softmax first)
// Target is class indices

class NLLLoss {
public:
    explicit NLLLoss(Reduction reduction = Reduction::MEAN, int64_t ignore_index = -100);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
    int64_t ignore_index_;
};

// ============================================================================
// CrossEntropyLoss - Combines LogSoftmax and NLLLoss
// ============================================================================
// L = -log(softmax(input)[class])
// Input: raw logits [N, C] or [N, C, ...] for spatial
// Target: class indices [N] or [N, ...] for spatial

class CrossEntropyLoss {
public:
    explicit CrossEntropyLoss(Reduction reduction = Reduction::MEAN,
                              int64_t ignore_index = -100,
                              float label_smoothing = 0.0f);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
    int64_t ignore_index_;
    float label_smoothing_;
};

// ============================================================================
// KLDivLoss - Kullback-Leibler Divergence Loss
// ============================================================================
// L = target * (log(target) - input)
// Input should be log-probabilities

class KLDivLoss {
public:
    explicit KLDivLoss(Reduction reduction = Reduction::MEAN, bool log_target = false);

    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return forward(input, target);
    }

    std::string to_string() const;

private:
    Reduction reduction_;
    bool log_target_;
};

// ============================================================================
// CosineEmbeddingLoss - Measures cosine distance between embeddings
// ============================================================================
// L = 1 - cos(x1, x2),           if y = 1
//   = max(0, cos(x1, x2) - margin), if y = -1

class CosineEmbeddingLoss {
public:
    explicit CosineEmbeddingLoss(Reduction reduction = Reduction::MEAN, float margin = 0.0f);

    Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& target);

    std::string to_string() const;

private:
    Reduction reduction_;
    float margin_;
};

// ============================================================================
// TripletMarginLoss - Triplet loss for metric learning
// ============================================================================
// L = max(d(a, p) - d(a, n) + margin, 0)
// where d is distance function (default: Euclidean)

class TripletMarginLoss {
public:
    explicit TripletMarginLoss(Reduction reduction = Reduction::MEAN,
                               float margin = 1.0f,
                               float p = 2.0f);

    Tensor forward(const Tensor& anchor, const Tensor& positive, const Tensor& negative);

    std::string to_string() const;

private:
    Reduction reduction_;
    float margin_;
    float p_;  // Norm degree
};

// ============================================================================
// Functional interface for loss functions
// ============================================================================

namespace functional {

Tensor mse_loss(const Tensor& input, const Tensor& target,
                Reduction reduction = Reduction::MEAN);

Tensor l1_loss(const Tensor& input, const Tensor& target,
               Reduction reduction = Reduction::MEAN);

Tensor smooth_l1_loss(const Tensor& input, const Tensor& target,
                      Reduction reduction = Reduction::MEAN,
                      float beta = 1.0f);

Tensor bce_loss(const Tensor& input, const Tensor& target,
                Reduction reduction = Reduction::MEAN);

Tensor bce_with_logits_loss(const Tensor& input, const Tensor& target,
                            Reduction reduction = Reduction::MEAN);

Tensor nll_loss(const Tensor& input, const Tensor& target,
                Reduction reduction = Reduction::MEAN,
                int64_t ignore_index = -100);

Tensor cross_entropy_loss(const Tensor& input, const Tensor& target,
                          Reduction reduction = Reduction::MEAN,
                          int64_t ignore_index = -100,
                          float label_smoothing = 0.0f);

Tensor kl_div_loss(const Tensor& input, const Tensor& target,
                   Reduction reduction = Reduction::MEAN,
                   bool log_target = false);

}  // namespace functional

}  // namespace pyflame::nn
