#pragma once

#include "pyflame/core/tensor.hpp"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace pyflame::optim {

// ============================================================================
// Optimizer Base Class
// ============================================================================

class Optimizer {
public:
    virtual ~Optimizer() = default;

    /// Perform a single optimization step
    virtual void step() = 0;

    /// Zero out all parameter gradients
    virtual void zero_grad();

    /// Get current learning rate
    float get_lr() const { return lr_; }

    /// Set learning rate
    void set_lr(float lr) { lr_ = lr; }

    /// Get the parameters being optimized
    const std::vector<Tensor*>& params() const { return params_; }

    /// Get optimizer state dictionary for checkpointing
    virtual std::unordered_map<std::string, Tensor> state_dict() const;

    /// Load optimizer state from dictionary
    virtual void load_state_dict(const std::unordered_map<std::string, Tensor>& state);

    /// Get string representation
    virtual std::string to_string() const = 0;

protected:
    Optimizer(std::vector<Tensor*> params, float lr);

    std::vector<Tensor*> params_;
    float lr_;
    int64_t step_count_ = 0;
};

// ============================================================================
// SGD - Stochastic Gradient Descent
// ============================================================================
// v = momentum * v + grad
// if nesterov:
//     param = param - lr * (grad + momentum * v)
// else:
//     param = param - lr * v

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params,
        float lr,
        float momentum = 0.0f,
        float dampening = 0.0f,
        float weight_decay = 0.0f,
        bool nesterov = false);

    void step() override;

    std::unordered_map<std::string, Tensor> state_dict() const override;
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state) override;

    std::string to_string() const override;

    // Accessors
    float momentum() const { return momentum_; }
    float dampening() const { return dampening_; }
    float weight_decay() const { return weight_decay_; }
    bool nesterov() const { return nesterov_; }

private:
    float momentum_;
    float dampening_;
    float weight_decay_;
    bool nesterov_;

    // Momentum buffers - one per parameter
    std::vector<Tensor> momentum_buffers_;
};

// ============================================================================
// Adam - Adaptive Moment Estimation
// ============================================================================
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// param = param - lr * m_hat / (sqrt(v_hat) + eps)

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params,
         float lr = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f,
         float weight_decay = 0.0f,
         bool amsgrad = false);

    void step() override;

    std::unordered_map<std::string, Tensor> state_dict() const override;
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state) override;

    std::string to_string() const override;

    // Accessors
    float beta1() const { return beta1_; }
    float beta2() const { return beta2_; }
    float eps() const { return eps_; }
    float weight_decay() const { return weight_decay_; }
    bool amsgrad() const { return amsgrad_; }

private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    bool amsgrad_;

    // First moment (mean of gradients)
    std::vector<Tensor> exp_avg_;
    // Second moment (mean of squared gradients)
    std::vector<Tensor> exp_avg_sq_;
    // Max of second moment (for amsgrad)
    std::vector<Tensor> max_exp_avg_sq_;
};

// ============================================================================
// AdamW - Adam with Decoupled Weight Decay
// ============================================================================
// Same as Adam but applies weight decay directly to weights
// rather than through the gradient

class AdamW : public Optimizer {
public:
    AdamW(std::vector<Tensor*> params,
          float lr = 0.001f,
          float beta1 = 0.9f,
          float beta2 = 0.999f,
          float eps = 1e-8f,
          float weight_decay = 0.01f,
          bool amsgrad = false);

    void step() override;

    std::unordered_map<std::string, Tensor> state_dict() const override;
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state) override;

    std::string to_string() const override;

    // Accessors
    float beta1() const { return beta1_; }
    float beta2() const { return beta2_; }
    float eps() const { return eps_; }
    float weight_decay() const { return weight_decay_; }
    bool amsgrad() const { return amsgrad_; }

private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    bool amsgrad_;

    std::vector<Tensor> exp_avg_;
    std::vector<Tensor> exp_avg_sq_;
    std::vector<Tensor> max_exp_avg_sq_;
};

// ============================================================================
// RMSprop - Root Mean Square Propagation
// ============================================================================
// v = alpha * v + (1 - alpha) * grad^2
// if centered:
//     g = alpha * g + (1 - alpha) * grad
//     v_hat = v - g^2
// else:
//     v_hat = v
// if momentum > 0:
//     buf = momentum * buf + grad / sqrt(v_hat + eps)
//     param = param - lr * buf
// else:
//     param = param - lr * grad / sqrt(v_hat + eps)

class RMSprop : public Optimizer {
public:
    RMSprop(std::vector<Tensor*> params,
            float lr = 0.01f,
            float alpha = 0.99f,
            float eps = 1e-8f,
            float weight_decay = 0.0f,
            float momentum = 0.0f,
            bool centered = false);

    void step() override;

    std::unordered_map<std::string, Tensor> state_dict() const override;
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state) override;

    std::string to_string() const override;

    // Accessors
    float alpha() const { return alpha_; }
    float eps() const { return eps_; }
    float weight_decay() const { return weight_decay_; }
    float momentum() const { return momentum_; }
    bool centered() const { return centered_; }

private:
    float alpha_;
    float eps_;
    float weight_decay_;
    float momentum_;
    bool centered_;

    std::vector<Tensor> square_avg_;      // Running average of squared gradients
    std::vector<Tensor> grad_avg_;        // For centered RMSprop
    std::vector<Tensor> momentum_buffer_; // For momentum
};

// ============================================================================
// Adagrad - Adaptive Gradient Algorithm
// ============================================================================
// state_sum = state_sum + grad^2
// param = param - lr * grad / (sqrt(state_sum) + eps)

class Adagrad : public Optimizer {
public:
    Adagrad(std::vector<Tensor*> params,
            float lr = 0.01f,
            float lr_decay = 0.0f,
            float weight_decay = 0.0f,
            float initial_accumulator_value = 0.0f,
            float eps = 1e-10f);

    void step() override;

    std::unordered_map<std::string, Tensor> state_dict() const override;
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state) override;

    std::string to_string() const override;

private:
    float lr_decay_;
    float weight_decay_;
    float initial_accumulator_value_;
    float eps_;

    std::vector<Tensor> sum_;  // Sum of squared gradients
};

// ============================================================================
// Helper: Create optimizer from module parameters
// ============================================================================

template<typename OptimizerT, typename... Args>
std::unique_ptr<OptimizerT> make_optimizer(
    const std::vector<Tensor*>& params,
    Args&&... args
) {
    return std::make_unique<OptimizerT>(params, std::forward<Args>(args)...);
}

}  // namespace pyflame::optim
