#pragma once

#include "pyflame/optim/optimizer.hpp"

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

namespace pyflame::optim {

// ============================================================================
// LRScheduler Base Class
// ============================================================================

class LRScheduler {
public:
    virtual ~LRScheduler() = default;

    /// Advance the scheduler by one step
    virtual void step() = 0;

    /// Get current learning rate
    float get_lr() const;

    /// Get current epoch/step count
    int64_t last_epoch() const { return last_epoch_; }

    /// Get base learning rate
    float base_lr() const { return base_lr_; }

    /// Get string representation
    virtual std::string to_string() const = 0;

protected:
    LRScheduler(Optimizer& optimizer, int64_t last_epoch = -1);

    /// Update the optimizer's learning rate
    void set_lr(float lr);

    Optimizer& optimizer_;
    float base_lr_;
    int64_t last_epoch_;
};

// ============================================================================
// StepLR - Decays LR by gamma every step_size epochs
// ============================================================================
// lr = base_lr * gamma^(epoch / step_size)

class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer,
           int64_t step_size,
           float gamma = 0.1f,
           int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    int64_t step_size() const { return step_size_; }
    float gamma() const { return gamma_; }

private:
    int64_t step_size_;
    float gamma_;
};

// ============================================================================
// MultiStepLR - Decays LR by gamma at specified milestones
// ============================================================================

class MultiStepLR : public LRScheduler {
public:
    MultiStepLR(Optimizer& optimizer,
                std::vector<int64_t> milestones,
                float gamma = 0.1f,
                int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    const std::vector<int64_t>& milestones() const { return milestones_; }
    float gamma() const { return gamma_; }

private:
    std::vector<int64_t> milestones_;
    float gamma_;
};

// ============================================================================
// ExponentialLR - Decays LR by gamma every epoch
// ============================================================================
// lr = base_lr * gamma^epoch

class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer& optimizer,
                  float gamma,
                  int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    float gamma() const { return gamma_; }

private:
    float gamma_;
};

// ============================================================================
// LinearLR - Linear learning rate warmup/decay
// ============================================================================
// lr = base_lr * (start_factor + (end_factor - start_factor) * (epoch / total_iters))

class LinearLR : public LRScheduler {
public:
    LinearLR(Optimizer& optimizer,
             float start_factor = 1.0f / 3.0f,
             float end_factor = 1.0f,
             int64_t total_iters = 5,
             int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    float start_factor() const { return start_factor_; }
    float end_factor() const { return end_factor_; }
    int64_t total_iters() const { return total_iters_; }

private:
    float start_factor_;
    float end_factor_;
    int64_t total_iters_;
};

// ============================================================================
// CosineAnnealingLR - Cosine annealing schedule
// ============================================================================
// lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer,
                      int64_t T_max,
                      float eta_min = 0.0f,
                      int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    int64_t T_max() const { return T_max_; }
    float eta_min() const { return eta_min_; }

private:
    int64_t T_max_;
    float eta_min_;
};

// ============================================================================
// CosineAnnealingWarmRestarts - Cosine annealing with warm restarts
// ============================================================================
// lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * T_cur / T_i)) / 2

class CosineAnnealingWarmRestarts : public LRScheduler {
public:
    CosineAnnealingWarmRestarts(Optimizer& optimizer,
                                 int64_t T_0,
                                 int64_t T_mult = 1,
                                 float eta_min = 0.0f,
                                 int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    int64_t T_0() const { return T_0_; }
    int64_t T_mult() const { return T_mult_; }
    float eta_min() const { return eta_min_; }

private:
    int64_t T_0_;
    int64_t T_mult_;
    float eta_min_;
    int64_t T_i_;      // Current cycle length
    int64_t T_cur_;    // Current position in cycle
};

// ============================================================================
// ReduceLROnPlateau - Reduce LR when metric stops improving
// ============================================================================

class ReduceLROnPlateau {
public:
    enum class Mode { MIN, MAX };

    ReduceLROnPlateau(Optimizer& optimizer,
                      Mode mode = Mode::MIN,
                      float factor = 0.1f,
                      int64_t patience = 10,
                      float threshold = 1e-4f,
                      int64_t cooldown = 0,
                      float min_lr = 0.0f);

    /// Call with validation metric
    void step(float metric);

    /// Get current learning rate
    float get_lr() const;

    std::string to_string() const;

    Mode mode() const { return mode_; }
    float factor() const { return factor_; }
    int64_t patience() const { return patience_; }

private:
    bool is_better(float current, float best) const;

    Optimizer& optimizer_;
    Mode mode_;
    float factor_;
    int64_t patience_;
    float threshold_;
    int64_t cooldown_;
    float min_lr_;

    float best_;
    int64_t num_bad_epochs_;
    int64_t cooldown_counter_;
};

// ============================================================================
// OneCycleLR - 1cycle learning rate policy
// ============================================================================
// Implements the 1cycle learning rate policy from Smith & Topin (2019)

class OneCycleLR : public LRScheduler {
public:
    enum class AnnealStrategy { COS, LINEAR };

    OneCycleLR(Optimizer& optimizer,
               float max_lr,
               int64_t total_steps,
               float pct_start = 0.3f,
               AnnealStrategy anneal_strategy = AnnealStrategy::COS,
               float div_factor = 25.0f,
               float final_div_factor = 1e4f,
               int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    float max_lr() const { return max_lr_; }
    int64_t total_steps() const { return total_steps_; }

private:
    float max_lr_;
    int64_t total_steps_;
    float pct_start_;
    AnnealStrategy anneal_strategy_;
    float div_factor_;
    float final_div_factor_;

    float initial_lr_;
    float min_lr_;
    int64_t step_up_;
    int64_t step_down_;
};

// ============================================================================
// WarmupLR - Warmup wrapper for any scheduler
// ============================================================================

class WarmupLR : public LRScheduler {
public:
    WarmupLR(Optimizer& optimizer,
             std::unique_ptr<LRScheduler> scheduler,
             int64_t warmup_steps,
             float warmup_init_lr = 0.0f,
             int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    int64_t warmup_steps() const { return warmup_steps_; }
    float warmup_init_lr() const { return warmup_init_lr_; }

private:
    std::unique_ptr<LRScheduler> scheduler_;
    int64_t warmup_steps_;
    float warmup_init_lr_;
};

// ============================================================================
// LambdaLR - Custom lambda function scheduler
// ============================================================================

class LambdaLR : public LRScheduler {
public:
    using LRLambda = std::function<float(int64_t)>;

    LambdaLR(Optimizer& optimizer,
             LRLambda lr_lambda,
             int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

private:
    LRLambda lr_lambda_;
};

// ============================================================================
// PolynomialLR - Polynomial decay scheduler
// ============================================================================
// lr = (base_lr - end_lr) * (1 - epoch/total_iters)^power + end_lr

class PolynomialLR : public LRScheduler {
public:
    PolynomialLR(Optimizer& optimizer,
                 int64_t total_iters,
                 float power = 1.0f,
                 float end_lr = 0.0f,
                 int64_t last_epoch = -1);

    void step() override;
    std::string to_string() const override;

    int64_t total_iters() const { return total_iters_; }
    float power() const { return power_; }
    float end_lr() const { return end_lr_; }

private:
    int64_t total_iters_;
    float power_;
    float end_lr_;
};

}  // namespace pyflame::optim
