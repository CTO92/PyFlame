#include "pyflame/nn/loss.hpp"
#include "pyflame/ir/op_type.hpp"
#include "pyflame/ir/node.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/core/tensor_impl.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace pyflame::nn {

// Helper function to apply reduction
static Tensor apply_reduction(const Tensor& loss, Reduction reduction) {
    switch (reduction) {
        case Reduction::NONE:
            return loss;
        case Reduction::MEAN:
            return loss.mean();
        case Reduction::SUM:
            return loss.sum();
        default:
            throw std::invalid_argument("Unknown reduction type");
    }
}

// Helper function to get reduction string
static std::string reduction_to_string(Reduction reduction) {
    switch (reduction) {
        case Reduction::NONE: return "none";
        case Reduction::MEAN: return "mean";
        case Reduction::SUM: return "sum";
        default: return "unknown";
    }
}

// ============================================================================
// MSELoss Implementation
// ============================================================================

MSELoss::MSELoss(Reduction reduction)
    : reduction_(reduction)
{
}

Tensor MSELoss::forward(const Tensor& input, const Tensor& target) {
    // L = (input - target)^2
    auto diff = input - target;
    auto sq_diff = diff * diff;
    return apply_reduction(sq_diff, reduction_);
}

std::string MSELoss::to_string() const {
    std::ostringstream oss;
    oss << "MSELoss(reduction=" << reduction_to_string(reduction_) << ")";
    return oss.str();
}

// ============================================================================
// L1Loss Implementation
// ============================================================================

L1Loss::L1Loss(Reduction reduction)
    : reduction_(reduction)
{
}

Tensor L1Loss::forward(const Tensor& input, const Tensor& target) {
    // L = |input - target|
    auto diff = input - target;
    auto abs_diff = abs(diff);
    return apply_reduction(abs_diff, reduction_);
}

std::string L1Loss::to_string() const {
    std::ostringstream oss;
    oss << "L1Loss(reduction=" << reduction_to_string(reduction_) << ")";
    return oss.str();
}

// ============================================================================
// SmoothL1Loss Implementation
// ============================================================================

SmoothL1Loss::SmoothL1Loss(Reduction reduction, float beta)
    : reduction_(reduction), beta_(beta)
{
    if (beta <= 0.0f) {
        throw std::invalid_argument("beta must be positive");
    }
}

Tensor SmoothL1Loss::forward(const Tensor& input, const Tensor& target) {
    // Huber loss:
    // L = 0.5 * x^2 / beta,  if |x| < beta
    //   = |x| - 0.5 * beta,  otherwise
    auto diff = input - target;
    auto abs_diff = abs(diff);

    // Create the piecewise function using where operation
    auto graph = input.graph();
    ir::TensorSpec out_spec(input.shape(), input.dtype(), input.layout());

    // Condition: |x| < beta
    auto condition_node = graph->create_op(ir::OpType::LESS,
        {abs_diff.node()}, out_spec);
    condition_node->set_attr("value", beta_);

    // Quadratic part: 0.5 * x^2 / beta
    auto sq_diff = diff * diff;
    auto quadratic = sq_diff * (0.5f / beta_);

    // Linear part: |x| - 0.5 * beta
    auto linear = abs_diff - (0.5f * beta_);

    // Use where to combine
    auto where_node = graph->create_op(ir::OpType::WHERE,
        {condition_node, quadratic.node(), linear.node()}, out_spec);

    auto impl = TensorImpl::from_node(graph, where_node);
    auto loss = Tensor::from_impl(impl);

    return apply_reduction(loss, reduction_);
}

std::string SmoothL1Loss::to_string() const {
    std::ostringstream oss;
    oss << "SmoothL1Loss(reduction=" << reduction_to_string(reduction_)
        << ", beta=" << beta_ << ")";
    return oss.str();
}

// ============================================================================
// HuberLoss Implementation
// ============================================================================

HuberLoss::HuberLoss(Reduction reduction, float delta)
    : reduction_(reduction), delta_(delta)
{
    if (delta <= 0.0f) {
        throw std::invalid_argument("delta must be positive");
    }
}

Tensor HuberLoss::forward(const Tensor& input, const Tensor& target) {
    // Same as SmoothL1Loss but with delta instead of beta
    auto diff = input - target;
    auto abs_diff = abs(diff);

    auto graph = input.graph();
    ir::TensorSpec out_spec(input.shape(), input.dtype(), input.layout());

    auto condition_node = graph->create_op(ir::OpType::LESS,
        {abs_diff.node()}, out_spec);
    condition_node->set_attr("value", delta_);

    auto sq_diff = diff * diff;
    auto quadratic = sq_diff * 0.5f;

    auto linear = abs_diff * delta_ - (0.5f * delta_ * delta_);

    auto where_node = graph->create_op(ir::OpType::WHERE,
        {condition_node, quadratic.node(), linear.node()}, out_spec);

    auto impl = TensorImpl::from_node(graph, where_node);
    auto loss = Tensor::from_impl(impl);

    return apply_reduction(loss, reduction_);
}

std::string HuberLoss::to_string() const {
    std::ostringstream oss;
    oss << "HuberLoss(reduction=" << reduction_to_string(reduction_)
        << ", delta=" << delta_ << ")";
    return oss.str();
}

// ============================================================================
// BCELoss Implementation
// ============================================================================

BCELoss::BCELoss(Reduction reduction)
    : reduction_(reduction)
{
}

Tensor BCELoss::forward(const Tensor& input, const Tensor& target) {
    // L = -[target * log(input) + (1 - target) * log(1 - input)]
    // Add small epsilon for numerical stability
    const float eps = 1e-7f;

    auto clamped_input = clamp(input, eps, 1.0f - eps);

    auto log_input = log(clamped_input);
    auto log_1_minus_input = log(Tensor::ones(input.shape()) - clamped_input);

    auto term1 = target * log_input;
    auto term2 = (Tensor::ones(target.shape()) - target) * log_1_minus_input;

    auto loss = -(term1 + term2);

    return apply_reduction(loss, reduction_);
}

std::string BCELoss::to_string() const {
    std::ostringstream oss;
    oss << "BCELoss(reduction=" << reduction_to_string(reduction_) << ")";
    return oss.str();
}

// ============================================================================
// BCEWithLogitsLoss Implementation
// ============================================================================

BCEWithLogitsLoss::BCEWithLogitsLoss(Reduction reduction)
    : reduction_(reduction)
{
}

Tensor BCEWithLogitsLoss::forward(const Tensor& input, const Tensor& target) {
    // Numerically stable BCE with logits
    // L = max(x, 0) - x*t + log(1 + exp(-|x|))

    auto graph = input.graph();
    ir::TensorSpec out_spec(input.shape(), input.dtype(), input.layout());

    // max(x, 0)
    auto zeros = Tensor::zeros(input.shape());
    auto max_x_0_node = graph->create_op(ir::OpType::MAXIMUM,
        {input.node(), zeros.node()}, out_spec);
    auto max_x_0 = Tensor::from_impl(TensorImpl::from_node(graph, max_x_0_node));

    // -x * t
    auto neg_x_t = -(input * target);

    // log(1 + exp(-|x|))
    auto abs_x = abs(input);
    auto neg_abs_x = -abs_x;
    auto exp_neg_abs = exp(neg_abs_x);
    auto one_plus_exp = Tensor::ones(input.shape()) + exp_neg_abs;
    auto log_term = log(one_plus_exp);

    auto loss = max_x_0 + neg_x_t + log_term;

    return apply_reduction(loss, reduction_);
}

std::string BCEWithLogitsLoss::to_string() const {
    std::ostringstream oss;
    oss << "BCEWithLogitsLoss(reduction=" << reduction_to_string(reduction_) << ")";
    return oss.str();
}

// ============================================================================
// NLLLoss Implementation
// ============================================================================

NLLLoss::NLLLoss(Reduction reduction, int64_t ignore_index)
    : reduction_(reduction), ignore_index_(ignore_index)
{
}

Tensor NLLLoss::forward(const Tensor& input, const Tensor& target) {
    // input: [N, C] or [N, C, d1, d2, ...] (log probabilities)
    // target: [N] or [N, d1, d2, ...] (class indices)
    // L = -input[n, target[n]]

    auto graph = input.graph();
    auto input_shape = input.shape();
    auto target_shape = target.shape();

    // Create NLL loss operation
    ir::TensorSpec out_spec(target_shape, input.dtype(), input.layout());

    auto nll_node = graph->create_op(ir::OpType::NLL_LOSS,
        {input.node(), target.node()}, out_spec);
    nll_node->set_attr("ignore_index", ignore_index_);

    auto impl = TensorImpl::from_node(graph, nll_node);
    auto loss = Tensor::from_impl(impl);

    return apply_reduction(loss, reduction_);
}

std::string NLLLoss::to_string() const {
    std::ostringstream oss;
    oss << "NLLLoss(reduction=" << reduction_to_string(reduction_);
    if (ignore_index_ != -100) {
        oss << ", ignore_index=" << ignore_index_;
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// CrossEntropyLoss Implementation
// ============================================================================

CrossEntropyLoss::CrossEntropyLoss(Reduction reduction, int64_t ignore_index,
                                   float label_smoothing)
    : reduction_(reduction)
    , ignore_index_(ignore_index)
    , label_smoothing_(label_smoothing)
{
    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        throw std::invalid_argument("label_smoothing must be in [0, 1]");
    }
}

Tensor CrossEntropyLoss::forward(const Tensor& input, const Tensor& target) {
    // input: [N, C] or [N, C, d1, d2, ...] (raw logits)
    // target: [N] or [N, d1, d2, ...] (class indices)
    // L = -log(softmax(input)[target])

    auto graph = input.graph();
    auto input_shape = input.shape();
    auto target_shape = target.shape();

    // Create cross entropy loss operation (combines log_softmax + nll_loss)
    ir::TensorSpec out_spec(target_shape, input.dtype(), input.layout());

    auto ce_node = graph->create_op(ir::OpType::CROSS_ENTROPY_LOSS,
        {input.node(), target.node()}, out_spec);
    ce_node->set_attr("ignore_index", ignore_index_);
    ce_node->set_attr("label_smoothing", label_smoothing_);

    auto impl = TensorImpl::from_node(graph, ce_node);
    auto loss = Tensor::from_impl(impl);

    return apply_reduction(loss, reduction_);
}

std::string CrossEntropyLoss::to_string() const {
    std::ostringstream oss;
    oss << "CrossEntropyLoss(reduction=" << reduction_to_string(reduction_);
    if (ignore_index_ != -100) {
        oss << ", ignore_index=" << ignore_index_;
    }
    if (label_smoothing_ > 0.0f) {
        oss << ", label_smoothing=" << label_smoothing_;
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// KLDivLoss Implementation
// ============================================================================

KLDivLoss::KLDivLoss(Reduction reduction, bool log_target)
    : reduction_(reduction), log_target_(log_target)
{
}

Tensor KLDivLoss::forward(const Tensor& input, const Tensor& target) {
    // input: log probabilities
    // target: probabilities (or log probabilities if log_target=true)
    // L = target * (log(target) - input)  [if not log_target]
    //   = exp(target) * (target - input)  [if log_target]

    Tensor loss;
    if (log_target_) {
        auto exp_target = exp(target);
        loss = exp_target * (target - input);
    } else {
        // Avoid log(0) by clamping
        const float eps = 1e-7f;
        auto safe_target = clamp(target, eps, 1.0f);
        auto log_target_val = log(safe_target);
        loss = target * (log_target_val - input);
    }

    return apply_reduction(loss, reduction_);
}

std::string KLDivLoss::to_string() const {
    std::ostringstream oss;
    oss << "KLDivLoss(reduction=" << reduction_to_string(reduction_);
    if (log_target_) {
        oss << ", log_target=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// CosineEmbeddingLoss Implementation
// ============================================================================

CosineEmbeddingLoss::CosineEmbeddingLoss(Reduction reduction, float margin)
    : reduction_(reduction), margin_(margin)
{
}

Tensor CosineEmbeddingLoss::forward(const Tensor& input1, const Tensor& input2,
                                     const Tensor& target) {
    // target: 1 (similar) or -1 (dissimilar)
    // L = 1 - cos(x1, x2),                  if y = 1
    //   = max(0, cos(x1, x2) - margin),     if y = -1

    // Compute cosine similarity
    auto dot = (input1 * input2).sum(-1);  // Sum along last dimension
    auto norm1 = sqrt((input1 * input1).sum(-1));
    auto norm2 = sqrt((input2 * input2).sum(-1));
    auto cos_sim = dot / (norm1 * norm2 + 1e-8f);

    // For y = 1: 1 - cos
    auto loss_similar = Tensor::ones(cos_sim.shape()) - cos_sim;

    // For y = -1: max(0, cos - margin)
    auto cos_minus_margin = cos_sim - margin_;
    auto graph = input1.graph();
    ir::TensorSpec out_spec(cos_sim.shape(), cos_sim.dtype(), cos_sim.layout());
    auto zeros = Tensor::zeros(cos_sim.shape());
    auto max_node = graph->create_op(ir::OpType::MAXIMUM,
        {cos_minus_margin.node(), zeros.node()}, out_spec);
    auto loss_dissimilar = Tensor::from_impl(TensorImpl::from_node(graph, max_node));

    // Select based on target
    // target = 1 -> use loss_similar
    // target = -1 -> use loss_dissimilar
    // Equivalent to: 0.5 * ((1 + target) * loss_similar + (1 - target) * loss_dissimilar)
    auto one = Tensor::ones(target.shape());
    auto pos_mask = (one + target) * 0.5f;
    auto neg_mask = (one - target) * 0.5f;

    auto loss = pos_mask * loss_similar + neg_mask * loss_dissimilar;

    return apply_reduction(loss, reduction_);
}

std::string CosineEmbeddingLoss::to_string() const {
    std::ostringstream oss;
    oss << "CosineEmbeddingLoss(reduction=" << reduction_to_string(reduction_)
        << ", margin=" << margin_ << ")";
    return oss.str();
}

// ============================================================================
// TripletMarginLoss Implementation
// ============================================================================

TripletMarginLoss::TripletMarginLoss(Reduction reduction, float margin, float p)
    : reduction_(reduction), margin_(margin), p_(p)
{
}

Tensor TripletMarginLoss::forward(const Tensor& anchor, const Tensor& positive,
                                   const Tensor& negative) {
    // L = max(d(a, p) - d(a, n) + margin, 0)
    // where d is p-norm distance

    // Compute distances
    auto diff_pos = anchor - positive;
    auto diff_neg = anchor - negative;

    Tensor dist_pos, dist_neg;

    if (p_ == 2.0f) {
        // Euclidean distance
        dist_pos = sqrt((diff_pos * diff_pos).sum(-1));
        dist_neg = sqrt((diff_neg * diff_neg).sum(-1));
    } else if (p_ == 1.0f) {
        // Manhattan distance
        dist_pos = abs(diff_pos).sum(-1);
        dist_neg = abs(diff_neg).sum(-1);
    } else {
        // General p-norm
        auto abs_diff_pos = abs(diff_pos);
        auto abs_diff_neg = abs(diff_neg);
        dist_pos = pow(pow(abs_diff_pos, p_).sum(-1), 1.0f / p_);
        dist_neg = pow(pow(abs_diff_neg, p_).sum(-1), 1.0f / p_);
    }

    // max(dist_pos - dist_neg + margin, 0)
    auto triplet_loss = dist_pos - dist_neg + margin_;
    auto graph = anchor.graph();
    ir::TensorSpec out_spec(triplet_loss.shape(), triplet_loss.dtype(), triplet_loss.layout());
    auto zeros = Tensor::zeros(triplet_loss.shape());
    auto max_node = graph->create_op(ir::OpType::MAXIMUM,
        {triplet_loss.node(), zeros.node()}, out_spec);
    auto loss = Tensor::from_impl(TensorImpl::from_node(graph, max_node));

    return apply_reduction(loss, reduction_);
}

std::string TripletMarginLoss::to_string() const {
    std::ostringstream oss;
    oss << "TripletMarginLoss(reduction=" << reduction_to_string(reduction_)
        << ", margin=" << margin_ << ", p=" << p_ << ")";
    return oss.str();
}

// ============================================================================
// Functional Interface Implementation
// ============================================================================

namespace functional {

Tensor mse_loss(const Tensor& input, const Tensor& target, Reduction reduction) {
    MSELoss loss(reduction);
    return loss.forward(input, target);
}

Tensor l1_loss(const Tensor& input, const Tensor& target, Reduction reduction) {
    L1Loss loss(reduction);
    return loss.forward(input, target);
}

Tensor smooth_l1_loss(const Tensor& input, const Tensor& target,
                      Reduction reduction, float beta) {
    SmoothL1Loss loss(reduction, beta);
    return loss.forward(input, target);
}

Tensor bce_loss(const Tensor& input, const Tensor& target, Reduction reduction) {
    BCELoss loss(reduction);
    return loss.forward(input, target);
}

Tensor bce_with_logits_loss(const Tensor& input, const Tensor& target,
                            Reduction reduction) {
    BCEWithLogitsLoss loss(reduction);
    return loss.forward(input, target);
}

Tensor nll_loss(const Tensor& input, const Tensor& target,
                Reduction reduction, int64_t ignore_index) {
    NLLLoss loss(reduction, ignore_index);
    return loss.forward(input, target);
}

Tensor cross_entropy_loss(const Tensor& input, const Tensor& target,
                          Reduction reduction, int64_t ignore_index,
                          float label_smoothing) {
    CrossEntropyLoss loss(reduction, ignore_index, label_smoothing);
    return loss.forward(input, target);
}

Tensor kl_div_loss(const Tensor& input, const Tensor& target,
                   Reduction reduction, bool log_target) {
    KLDivLoss loss(reduction, log_target);
    return loss.forward(input, target);
}

}  // namespace functional

}  // namespace pyflame::nn
