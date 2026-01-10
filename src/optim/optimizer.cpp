#include "pyflame/optim/optimizer.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace pyflame::optim {

// ============================================================================
// Optimizer Base Implementation
// ============================================================================

Optimizer::Optimizer(std::vector<Tensor*> params, float lr)
    : params_(std::move(params)), lr_(lr)
{
    if (lr < 0.0f) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
}

void Optimizer::zero_grad() {
    for (auto* param : params_) {
        if (param && param->grad().numel() > 0) {
            param->zero_grad();
        }
    }
}

std::unordered_map<std::string, Tensor> Optimizer::state_dict() const {
    std::unordered_map<std::string, Tensor> state;
    // Base class stores step count
    state["step"] = Tensor::full({1}, static_cast<float>(step_count_));
    return state;
}

void Optimizer::load_state_dict(const std::unordered_map<std::string, Tensor>& state) {
    auto it = state.find("step");
    if (it != state.end()) {
        step_count_ = static_cast<int64_t>(it->second.data<float>()[0]);
    }
}

// ============================================================================
// SGD Implementation
// ============================================================================

SGD::SGD(std::vector<Tensor*> params, float lr, float momentum,
         float dampening, float weight_decay, bool nesterov)
    : Optimizer(std::move(params), lr)
    , momentum_(momentum)
    , dampening_(dampening)
    , weight_decay_(weight_decay)
    , nesterov_(nesterov)
{
    if (momentum < 0.0f) {
        throw std::invalid_argument("Momentum must be non-negative");
    }
    if (nesterov && (momentum <= 0.0f || dampening != 0.0f)) {
        throw std::invalid_argument(
            "Nesterov momentum requires momentum > 0 and dampening = 0");
    }

    // Initialize momentum buffers (lazily, on first step)
    momentum_buffers_.resize(params_.size());
}

void SGD::step() {
    step_count_++;

    for (size_t i = 0; i < params_.size(); ++i) {
        auto* param = params_[i];
        if (!param) continue;

        auto grad = param->grad();
        if (grad.numel() == 0) continue;

        // Apply weight decay (L2 regularization)
        if (weight_decay_ != 0.0f) {
            grad = grad + (*param) * weight_decay_;
        }

        // Apply momentum
        if (momentum_ != 0.0f) {
            if (momentum_buffers_[i].numel() == 0) {
                // First step: initialize buffer with gradient
                momentum_buffers_[i] = grad.clone();
            } else {
                // buf = momentum * buf + (1 - dampening) * grad
                momentum_buffers_[i] = momentum_buffers_[i] * momentum_ +
                                       grad * (1.0f - dampening_);
            }

            if (nesterov_) {
                // Nesterov: use grad + momentum * buf
                grad = grad + momentum_buffers_[i] * momentum_;
            } else {
                grad = momentum_buffers_[i];
            }
        }

        // Update parameter
        *param = *param - grad * lr_;
    }
}

std::unordered_map<std::string, Tensor> SGD::state_dict() const {
    auto state = Optimizer::state_dict();

    // Save momentum buffers
    for (size_t i = 0; i < momentum_buffers_.size(); ++i) {
        if (momentum_buffers_[i].numel() > 0) {
            state["momentum_buffer_" + std::to_string(i)] = momentum_buffers_[i];
        }
    }

    return state;
}

void SGD::load_state_dict(const std::unordered_map<std::string, Tensor>& state) {
    Optimizer::load_state_dict(state);

    for (size_t i = 0; i < momentum_buffers_.size(); ++i) {
        auto key = "momentum_buffer_" + std::to_string(i);
        auto it = state.find(key);
        if (it != state.end()) {
            momentum_buffers_[i] = it->second;
        }
    }
}

std::string SGD::to_string() const {
    std::ostringstream oss;
    oss << "SGD(lr=" << lr_;
    if (momentum_ != 0.0f) {
        oss << ", momentum=" << momentum_;
    }
    if (dampening_ != 0.0f) {
        oss << ", dampening=" << dampening_;
    }
    if (weight_decay_ != 0.0f) {
        oss << ", weight_decay=" << weight_decay_;
    }
    if (nesterov_) {
        oss << ", nesterov=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// Adam Implementation
// ============================================================================

Adam::Adam(std::vector<Tensor*> params, float lr, float beta1, float beta2,
           float eps, float weight_decay, bool amsgrad)
    : Optimizer(std::move(params), lr)
    , beta1_(beta1)
    , beta2_(beta2)
    , eps_(eps)
    , weight_decay_(weight_decay)
    , amsgrad_(amsgrad)
{
    if (beta1 < 0.0f || beta1 >= 1.0f) {
        throw std::invalid_argument("beta1 must be in [0, 1)");
    }
    if (beta2 < 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument("beta2 must be in [0, 1)");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("eps must be positive");
    }

    // Initialize state buffers
    exp_avg_.resize(params_.size());
    exp_avg_sq_.resize(params_.size());
    if (amsgrad_) {
        max_exp_avg_sq_.resize(params_.size());
    }
}

void Adam::step() {
    step_count_++;

    // Bias correction terms
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

    for (size_t i = 0; i < params_.size(); ++i) {
        auto* param = params_[i];
        if (!param) continue;

        auto grad = param->grad();
        if (grad.numel() == 0) continue;

        // Apply L2 regularization to gradient (Adam style)
        if (weight_decay_ != 0.0f) {
            grad = grad + (*param) * weight_decay_;
        }

        // Initialize state on first step
        if (exp_avg_[i].numel() == 0) {
            exp_avg_[i] = Tensor::zeros(param->shape());
            exp_avg_sq_[i] = Tensor::zeros(param->shape());
            if (amsgrad_) {
                max_exp_avg_sq_[i] = Tensor::zeros(param->shape());
            }
        }

        // Update biased first moment estimate
        // m = beta1 * m + (1 - beta1) * grad
        exp_avg_[i] = exp_avg_[i] * beta1_ + grad * (1.0f - beta1_);

        // Update biased second raw moment estimate
        // v = beta2 * v + (1 - beta2) * grad^2
        exp_avg_sq_[i] = exp_avg_sq_[i] * beta2_ + (grad * grad) * (1.0f - beta2_);

        Tensor denom;
        if (amsgrad_) {
            // Use maximum of all second moments
            max_exp_avg_sq_[i] = maximum(max_exp_avg_sq_[i], exp_avg_sq_[i]);
            // denom = sqrt(max_v / bias_correction2) + eps
            denom = sqrt(max_exp_avg_sq_[i] / bias_correction2) + eps_;
        } else {
            // denom = sqrt(v / bias_correction2) + eps
            denom = sqrt(exp_avg_sq_[i] / bias_correction2) + eps_;
        }

        // Compute step size with bias correction
        float step_size = lr_ / bias_correction1;

        // Update parameter
        // param = param - step_size * m / denom
        *param = *param - (exp_avg_[i] / denom) * step_size;
    }
}

std::unordered_map<std::string, Tensor> Adam::state_dict() const {
    auto state = Optimizer::state_dict();

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        if (exp_avg_[i].numel() > 0) {
            state["exp_avg_" + std::to_string(i)] = exp_avg_[i];
            state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i];
            if (amsgrad_ && max_exp_avg_sq_[i].numel() > 0) {
                state["max_exp_avg_sq_" + std::to_string(i)] = max_exp_avg_sq_[i];
            }
        }
    }

    return state;
}

void Adam::load_state_dict(const std::unordered_map<std::string, Tensor>& state) {
    Optimizer::load_state_dict(state);

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        auto key1 = "exp_avg_" + std::to_string(i);
        auto key2 = "exp_avg_sq_" + std::to_string(i);
        auto it1 = state.find(key1);
        auto it2 = state.find(key2);
        if (it1 != state.end()) exp_avg_[i] = it1->second;
        if (it2 != state.end()) exp_avg_sq_[i] = it2->second;

        if (amsgrad_) {
            auto key3 = "max_exp_avg_sq_" + std::to_string(i);
            auto it3 = state.find(key3);
            if (it3 != state.end()) max_exp_avg_sq_[i] = it3->second;
        }
    }
}

std::string Adam::to_string() const {
    std::ostringstream oss;
    oss << "Adam(lr=" << lr_
        << ", betas=(" << beta1_ << ", " << beta2_ << ")"
        << ", eps=" << eps_;
    if (weight_decay_ != 0.0f) {
        oss << ", weight_decay=" << weight_decay_;
    }
    if (amsgrad_) {
        oss << ", amsgrad=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// AdamW Implementation
// ============================================================================

AdamW::AdamW(std::vector<Tensor*> params, float lr, float beta1, float beta2,
             float eps, float weight_decay, bool amsgrad)
    : Optimizer(std::move(params), lr)
    , beta1_(beta1)
    , beta2_(beta2)
    , eps_(eps)
    , weight_decay_(weight_decay)
    , amsgrad_(amsgrad)
{
    if (beta1 < 0.0f || beta1 >= 1.0f) {
        throw std::invalid_argument("beta1 must be in [0, 1)");
    }
    if (beta2 < 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument("beta2 must be in [0, 1)");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("eps must be positive");
    }

    exp_avg_.resize(params_.size());
    exp_avg_sq_.resize(params_.size());
    if (amsgrad_) {
        max_exp_avg_sq_.resize(params_.size());
    }
}

void AdamW::step() {
    step_count_++;

    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

    for (size_t i = 0; i < params_.size(); ++i) {
        auto* param = params_[i];
        if (!param) continue;

        auto grad = param->grad();
        if (grad.numel() == 0) continue;

        // Decoupled weight decay - apply directly to weights
        if (weight_decay_ != 0.0f) {
            *param = *param * (1.0f - lr_ * weight_decay_);
        }

        // Initialize state on first step
        if (exp_avg_[i].numel() == 0) {
            exp_avg_[i] = Tensor::zeros(param->shape());
            exp_avg_sq_[i] = Tensor::zeros(param->shape());
            if (amsgrad_) {
                max_exp_avg_sq_[i] = Tensor::zeros(param->shape());
            }
        }

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * beta1_ + grad * (1.0f - beta1_);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * beta2_ + (grad * grad) * (1.0f - beta2_);

        Tensor denom;
        if (amsgrad_) {
            max_exp_avg_sq_[i] = maximum(max_exp_avg_sq_[i], exp_avg_sq_[i]);
            denom = sqrt(max_exp_avg_sq_[i] / bias_correction2) + eps_;
        } else {
            denom = sqrt(exp_avg_sq_[i] / bias_correction2) + eps_;
        }

        float step_size = lr_ / bias_correction1;
        *param = *param - (exp_avg_[i] / denom) * step_size;
    }
}

std::unordered_map<std::string, Tensor> AdamW::state_dict() const {
    auto state = Optimizer::state_dict();

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        if (exp_avg_[i].numel() > 0) {
            state["exp_avg_" + std::to_string(i)] = exp_avg_[i];
            state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i];
            if (amsgrad_ && max_exp_avg_sq_[i].numel() > 0) {
                state["max_exp_avg_sq_" + std::to_string(i)] = max_exp_avg_sq_[i];
            }
        }
    }

    return state;
}

void AdamW::load_state_dict(const std::unordered_map<std::string, Tensor>& state) {
    Optimizer::load_state_dict(state);

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        auto key1 = "exp_avg_" + std::to_string(i);
        auto key2 = "exp_avg_sq_" + std::to_string(i);
        auto it1 = state.find(key1);
        auto it2 = state.find(key2);
        if (it1 != state.end()) exp_avg_[i] = it1->second;
        if (it2 != state.end()) exp_avg_sq_[i] = it2->second;

        if (amsgrad_) {
            auto key3 = "max_exp_avg_sq_" + std::to_string(i);
            auto it3 = state.find(key3);
            if (it3 != state.end()) max_exp_avg_sq_[i] = it3->second;
        }
    }
}

std::string AdamW::to_string() const {
    std::ostringstream oss;
    oss << "AdamW(lr=" << lr_
        << ", betas=(" << beta1_ << ", " << beta2_ << ")"
        << ", eps=" << eps_
        << ", weight_decay=" << weight_decay_;
    if (amsgrad_) {
        oss << ", amsgrad=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// RMSprop Implementation
// ============================================================================

RMSprop::RMSprop(std::vector<Tensor*> params, float lr, float alpha, float eps,
                 float weight_decay, float momentum, bool centered)
    : Optimizer(std::move(params), lr)
    , alpha_(alpha)
    , eps_(eps)
    , weight_decay_(weight_decay)
    , momentum_(momentum)
    , centered_(centered)
{
    if (alpha < 0.0f || alpha >= 1.0f) {
        throw std::invalid_argument("alpha must be in [0, 1)");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("eps must be positive");
    }

    square_avg_.resize(params_.size());
    if (centered_) {
        grad_avg_.resize(params_.size());
    }
    if (momentum_ > 0.0f) {
        momentum_buffer_.resize(params_.size());
    }
}

void RMSprop::step() {
    step_count_++;

    for (size_t i = 0; i < params_.size(); ++i) {
        auto* param = params_[i];
        if (!param) continue;

        auto grad = param->grad();
        if (grad.numel() == 0) continue;

        // Apply weight decay
        if (weight_decay_ != 0.0f) {
            grad = grad + (*param) * weight_decay_;
        }

        // Initialize state on first step
        if (square_avg_[i].numel() == 0) {
            square_avg_[i] = Tensor::zeros(param->shape());
            if (centered_) {
                grad_avg_[i] = Tensor::zeros(param->shape());
            }
            if (momentum_ > 0.0f) {
                momentum_buffer_[i] = Tensor::zeros(param->shape());
            }
        }

        // Update running average of squared gradients
        // v = alpha * v + (1 - alpha) * grad^2
        square_avg_[i] = square_avg_[i] * alpha_ + (grad * grad) * (1.0f - alpha_);

        Tensor avg;
        if (centered_) {
            // Update running average of gradients
            grad_avg_[i] = grad_avg_[i] * alpha_ + grad * (1.0f - alpha_);
            // avg = v - g^2
            avg = square_avg_[i] - grad_avg_[i] * grad_avg_[i];
        } else {
            avg = square_avg_[i];
        }

        // Compute update
        auto denom = sqrt(avg) + eps_;
        auto update = grad / denom;

        if (momentum_ > 0.0f) {
            momentum_buffer_[i] = momentum_buffer_[i] * momentum_ + update;
            *param = *param - momentum_buffer_[i] * lr_;
        } else {
            *param = *param - update * lr_;
        }
    }
}

std::unordered_map<std::string, Tensor> RMSprop::state_dict() const {
    auto state = Optimizer::state_dict();

    for (size_t i = 0; i < square_avg_.size(); ++i) {
        if (square_avg_[i].numel() > 0) {
            state["square_avg_" + std::to_string(i)] = square_avg_[i];
            if (centered_ && grad_avg_[i].numel() > 0) {
                state["grad_avg_" + std::to_string(i)] = grad_avg_[i];
            }
            if (momentum_ > 0.0f && momentum_buffer_[i].numel() > 0) {
                state["momentum_buffer_" + std::to_string(i)] = momentum_buffer_[i];
            }
        }
    }

    return state;
}

void RMSprop::load_state_dict(const std::unordered_map<std::string, Tensor>& state) {
    Optimizer::load_state_dict(state);

    for (size_t i = 0; i < square_avg_.size(); ++i) {
        auto key1 = "square_avg_" + std::to_string(i);
        auto it1 = state.find(key1);
        if (it1 != state.end()) square_avg_[i] = it1->second;

        if (centered_) {
            auto key2 = "grad_avg_" + std::to_string(i);
            auto it2 = state.find(key2);
            if (it2 != state.end()) grad_avg_[i] = it2->second;
        }

        if (momentum_ > 0.0f) {
            auto key3 = "momentum_buffer_" + std::to_string(i);
            auto it3 = state.find(key3);
            if (it3 != state.end()) momentum_buffer_[i] = it3->second;
        }
    }
}

std::string RMSprop::to_string() const {
    std::ostringstream oss;
    oss << "RMSprop(lr=" << lr_
        << ", alpha=" << alpha_
        << ", eps=" << eps_;
    if (weight_decay_ != 0.0f) {
        oss << ", weight_decay=" << weight_decay_;
    }
    if (momentum_ != 0.0f) {
        oss << ", momentum=" << momentum_;
    }
    if (centered_) {
        oss << ", centered=True";
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// Adagrad Implementation
// ============================================================================

Adagrad::Adagrad(std::vector<Tensor*> params, float lr, float lr_decay,
                 float weight_decay, float initial_accumulator_value, float eps)
    : Optimizer(std::move(params), lr)
    , lr_decay_(lr_decay)
    , weight_decay_(weight_decay)
    , initial_accumulator_value_(initial_accumulator_value)
    , eps_(eps)
{
    if (initial_accumulator_value < 0.0f) {
        throw std::invalid_argument("initial_accumulator_value must be non-negative");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("eps must be positive");
    }

    sum_.resize(params_.size());
}

void Adagrad::step() {
    step_count_++;

    // Compute decayed learning rate
    float clr = lr_ / (1.0f + (step_count_ - 1) * lr_decay_);

    for (size_t i = 0; i < params_.size(); ++i) {
        auto* param = params_[i];
        if (!param) continue;

        auto grad = param->grad();
        if (grad.numel() == 0) continue;

        // Apply weight decay
        if (weight_decay_ != 0.0f) {
            grad = grad + (*param) * weight_decay_;
        }

        // Initialize state on first step
        if (sum_[i].numel() == 0) {
            sum_[i] = Tensor::full(param->shape(), initial_accumulator_value_);
        }

        // Update sum of squared gradients
        sum_[i] = sum_[i] + grad * grad;

        // Update parameter
        auto std = sqrt(sum_[i]) + eps_;
        *param = *param - (grad / std) * clr;
    }
}

std::unordered_map<std::string, Tensor> Adagrad::state_dict() const {
    auto state = Optimizer::state_dict();

    for (size_t i = 0; i < sum_.size(); ++i) {
        if (sum_[i].numel() > 0) {
            state["sum_" + std::to_string(i)] = sum_[i];
        }
    }

    return state;
}

void Adagrad::load_state_dict(const std::unordered_map<std::string, Tensor>& state) {
    Optimizer::load_state_dict(state);

    for (size_t i = 0; i < sum_.size(); ++i) {
        auto key = "sum_" + std::to_string(i);
        auto it = state.find(key);
        if (it != state.end()) sum_[i] = it->second;
    }
}

std::string Adagrad::to_string() const {
    std::ostringstream oss;
    oss << "Adagrad(lr=" << lr_;
    if (lr_decay_ != 0.0f) {
        oss << ", lr_decay=" << lr_decay_;
    }
    if (weight_decay_ != 0.0f) {
        oss << ", weight_decay=" << weight_decay_;
    }
    if (initial_accumulator_value_ != 0.0f) {
        oss << ", initial_accumulator_value=" << initial_accumulator_value_;
    }
    oss << ", eps=" << eps_ << ")";
    return oss.str();
}

}  // namespace pyflame::optim
