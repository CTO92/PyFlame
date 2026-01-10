#include "pyflame/optim/lr_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace pyflame::optim {

// ============================================================================
// LRScheduler Base Implementation
// ============================================================================

LRScheduler::LRScheduler(Optimizer& optimizer, int64_t last_epoch)
    : optimizer_(optimizer)
    , base_lr_(optimizer.get_lr())
    , last_epoch_(last_epoch)
{
}

float LRScheduler::get_lr() const {
    return optimizer_.get_lr();
}

void LRScheduler::set_lr(float lr) {
    optimizer_.set_lr(lr);
}

// ============================================================================
// StepLR Implementation
// ============================================================================

StepLR::StepLR(Optimizer& optimizer, int64_t step_size, float gamma, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , step_size_(step_size)
    , gamma_(gamma)
{
    if (step_size <= 0) {
        throw std::invalid_argument("step_size must be positive");
    }
    if (gamma <= 0.0f || gamma > 1.0f) {
        throw std::invalid_argument("gamma must be in (0, 1]");
    }
}

void StepLR::step() {
    last_epoch_++;

    int64_t num_decays = last_epoch_ / step_size_;
    float lr = base_lr_ * std::pow(gamma_, static_cast<float>(num_decays));
    set_lr(lr);
}

std::string StepLR::to_string() const {
    std::ostringstream oss;
    oss << "StepLR(step_size=" << step_size_ << ", gamma=" << gamma_ << ")";
    return oss.str();
}

// ============================================================================
// MultiStepLR Implementation
// ============================================================================

MultiStepLR::MultiStepLR(Optimizer& optimizer, std::vector<int64_t> milestones,
                         float gamma, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , milestones_(std::move(milestones))
    , gamma_(gamma)
{
    // Sort milestones
    std::sort(milestones_.begin(), milestones_.end());

    if (gamma <= 0.0f || gamma > 1.0f) {
        throw std::invalid_argument("gamma must be in (0, 1]");
    }
}

void MultiStepLR::step() {
    last_epoch_++;

    // Count how many milestones we've passed
    int64_t num_decays = 0;
    for (int64_t milestone : milestones_) {
        if (last_epoch_ >= milestone) {
            num_decays++;
        }
    }

    float lr = base_lr_ * std::pow(gamma_, static_cast<float>(num_decays));
    set_lr(lr);
}

std::string MultiStepLR::to_string() const {
    std::ostringstream oss;
    oss << "MultiStepLR(milestones=[";
    for (size_t i = 0; i < milestones_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << milestones_[i];
    }
    oss << "], gamma=" << gamma_ << ")";
    return oss.str();
}

// ============================================================================
// ExponentialLR Implementation
// ============================================================================

ExponentialLR::ExponentialLR(Optimizer& optimizer, float gamma, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , gamma_(gamma)
{
    if (gamma <= 0.0f || gamma > 1.0f) {
        throw std::invalid_argument("gamma must be in (0, 1]");
    }
}

void ExponentialLR::step() {
    last_epoch_++;
    float lr = base_lr_ * std::pow(gamma_, static_cast<float>(last_epoch_));
    set_lr(lr);
}

std::string ExponentialLR::to_string() const {
    std::ostringstream oss;
    oss << "ExponentialLR(gamma=" << gamma_ << ")";
    return oss.str();
}

// ============================================================================
// LinearLR Implementation
// ============================================================================

LinearLR::LinearLR(Optimizer& optimizer, float start_factor, float end_factor,
                   int64_t total_iters, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , start_factor_(start_factor)
    , end_factor_(end_factor)
    , total_iters_(total_iters)
{
    if (start_factor < 0.0f || start_factor > 1.0f) {
        throw std::invalid_argument("start_factor must be in [0, 1]");
    }
    if (end_factor < 0.0f) {
        throw std::invalid_argument("end_factor must be non-negative");
    }
    if (total_iters <= 0) {
        throw std::invalid_argument("total_iters must be positive");
    }
}

void LinearLR::step() {
    last_epoch_++;

    float factor;
    if (last_epoch_ >= total_iters_) {
        factor = end_factor_;
    } else {
        float progress = static_cast<float>(last_epoch_) / static_cast<float>(total_iters_);
        factor = start_factor_ + (end_factor_ - start_factor_) * progress;
    }

    set_lr(base_lr_ * factor);
}

std::string LinearLR::to_string() const {
    std::ostringstream oss;
    oss << "LinearLR(start_factor=" << start_factor_
        << ", end_factor=" << end_factor_
        << ", total_iters=" << total_iters_ << ")";
    return oss.str();
}

// ============================================================================
// CosineAnnealingLR Implementation
// ============================================================================

CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer, int64_t T_max,
                                     float eta_min, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , T_max_(T_max)
    , eta_min_(eta_min)
{
    if (T_max <= 0) {
        throw std::invalid_argument("T_max must be positive");
    }
}

void CosineAnnealingLR::step() {
    last_epoch_++;

    const float pi = 3.14159265358979323846f;
    float lr = eta_min_ + (base_lr_ - eta_min_) *
               (1.0f + std::cos(pi * static_cast<float>(last_epoch_) / static_cast<float>(T_max_))) / 2.0f;
    set_lr(lr);
}

std::string CosineAnnealingLR::to_string() const {
    std::ostringstream oss;
    oss << "CosineAnnealingLR(T_max=" << T_max_ << ", eta_min=" << eta_min_ << ")";
    return oss.str();
}

// ============================================================================
// CosineAnnealingWarmRestarts Implementation
// ============================================================================

CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(
    Optimizer& optimizer, int64_t T_0, int64_t T_mult, float eta_min, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , T_0_(T_0)
    , T_mult_(T_mult)
    , eta_min_(eta_min)
    , T_i_(T_0)
    , T_cur_(0)
{
    if (T_0 <= 0) {
        throw std::invalid_argument("T_0 must be positive");
    }
    if (T_mult < 1) {
        throw std::invalid_argument("T_mult must be >= 1");
    }
}

void CosineAnnealingWarmRestarts::step() {
    last_epoch_++;
    T_cur_++;

    // Check if we need to restart
    if (T_cur_ >= T_i_) {
        T_cur_ = 0;
        T_i_ *= T_mult_;
    }

    const float pi = 3.14159265358979323846f;
    float lr = eta_min_ + (base_lr_ - eta_min_) *
               (1.0f + std::cos(pi * static_cast<float>(T_cur_) / static_cast<float>(T_i_))) / 2.0f;
    set_lr(lr);
}

std::string CosineAnnealingWarmRestarts::to_string() const {
    std::ostringstream oss;
    oss << "CosineAnnealingWarmRestarts(T_0=" << T_0_
        << ", T_mult=" << T_mult_
        << ", eta_min=" << eta_min_ << ")";
    return oss.str();
}

// ============================================================================
// ReduceLROnPlateau Implementation
// ============================================================================

ReduceLROnPlateau::ReduceLROnPlateau(Optimizer& optimizer, Mode mode, float factor,
                                     int64_t patience, float threshold,
                                     int64_t cooldown, float min_lr)
    : optimizer_(optimizer)
    , mode_(mode)
    , factor_(factor)
    , patience_(patience)
    , threshold_(threshold)
    , cooldown_(cooldown)
    , min_lr_(min_lr)
    , num_bad_epochs_(0)
    , cooldown_counter_(0)
{
    if (factor >= 1.0f) {
        throw std::invalid_argument("factor must be < 1.0");
    }
    if (patience < 0) {
        throw std::invalid_argument("patience must be non-negative");
    }

    best_ = (mode == Mode::MIN) ? std::numeric_limits<float>::infinity()
                                 : -std::numeric_limits<float>::infinity();
}

bool ReduceLROnPlateau::is_better(float current, float best) const {
    if (mode_ == Mode::MIN) {
        return current < best - threshold_;
    } else {
        return current > best + threshold_;
    }
}

void ReduceLROnPlateau::step(float metric) {
    // Handle cooldown
    if (cooldown_counter_ > 0) {
        cooldown_counter_--;
        num_bad_epochs_ = 0;
    }

    if (is_better(metric, best_)) {
        best_ = metric;
        num_bad_epochs_ = 0;
    } else {
        num_bad_epochs_++;
    }

    // Reduce LR if patience exceeded
    if (num_bad_epochs_ > patience_) {
        float old_lr = optimizer_.get_lr();
        float new_lr = std::max(old_lr * factor_, min_lr_);

        if (old_lr - new_lr > 1e-8f) {
            optimizer_.set_lr(new_lr);
            cooldown_counter_ = cooldown_;
            num_bad_epochs_ = 0;
        }
    }
}

float ReduceLROnPlateau::get_lr() const {
    return optimizer_.get_lr();
}

std::string ReduceLROnPlateau::to_string() const {
    std::ostringstream oss;
    oss << "ReduceLROnPlateau(mode=" << (mode_ == Mode::MIN ? "min" : "max")
        << ", factor=" << factor_
        << ", patience=" << patience_
        << ", threshold=" << threshold_
        << ", cooldown=" << cooldown_
        << ", min_lr=" << min_lr_ << ")";
    return oss.str();
}

// ============================================================================
// OneCycleLR Implementation
// ============================================================================

OneCycleLR::OneCycleLR(Optimizer& optimizer, float max_lr, int64_t total_steps,
                       float pct_start, AnnealStrategy anneal_strategy,
                       float div_factor, float final_div_factor, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , max_lr_(max_lr)
    , total_steps_(total_steps)
    , pct_start_(pct_start)
    , anneal_strategy_(anneal_strategy)
    , div_factor_(div_factor)
    , final_div_factor_(final_div_factor)
{
    if (total_steps <= 0) {
        throw std::invalid_argument("total_steps must be positive");
    }
    if (pct_start < 0.0f || pct_start > 1.0f) {
        throw std::invalid_argument("pct_start must be in [0, 1]");
    }

    initial_lr_ = max_lr_ / div_factor_;
    min_lr_ = initial_lr_ / final_div_factor_;
    step_up_ = static_cast<int64_t>(total_steps_ * pct_start_);
    step_down_ = total_steps_ - step_up_;

    // Set initial LR
    optimizer_.set_lr(initial_lr_);
}

void OneCycleLR::step() {
    last_epoch_++;

    float lr;
    if (last_epoch_ <= step_up_) {
        // Warmup phase: linear increase from initial_lr to max_lr
        float progress = static_cast<float>(last_epoch_) / static_cast<float>(step_up_);
        lr = initial_lr_ + (max_lr_ - initial_lr_) * progress;
    } else {
        // Annealing phase
        int64_t step_in_phase = last_epoch_ - step_up_;
        float progress = static_cast<float>(step_in_phase) / static_cast<float>(step_down_);

        if (anneal_strategy_ == AnnealStrategy::COS) {
            const float pi = 3.14159265358979323846f;
            lr = min_lr_ + (max_lr_ - min_lr_) * (1.0f + std::cos(pi * progress)) / 2.0f;
        } else {
            // Linear annealing
            lr = max_lr_ - (max_lr_ - min_lr_) * progress;
        }
    }

    set_lr(lr);
}

std::string OneCycleLR::to_string() const {
    std::ostringstream oss;
    oss << "OneCycleLR(max_lr=" << max_lr_
        << ", total_steps=" << total_steps_
        << ", pct_start=" << pct_start_
        << ", anneal_strategy=" << (anneal_strategy_ == AnnealStrategy::COS ? "cos" : "linear")
        << ", div_factor=" << div_factor_
        << ", final_div_factor=" << final_div_factor_ << ")";
    return oss.str();
}

// ============================================================================
// WarmupLR Implementation
// ============================================================================

WarmupLR::WarmupLR(Optimizer& optimizer, std::unique_ptr<LRScheduler> scheduler,
                   int64_t warmup_steps, float warmup_init_lr, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , scheduler_(std::move(scheduler))
    , warmup_steps_(warmup_steps)
    , warmup_init_lr_(warmup_init_lr)
{
    if (warmup_steps < 0) {
        throw std::invalid_argument("warmup_steps must be non-negative");
    }

    // Set initial warmup LR
    if (warmup_steps_ > 0) {
        optimizer_.set_lr(warmup_init_lr_);
    }
}

void WarmupLR::step() {
    last_epoch_++;

    if (last_epoch_ <= warmup_steps_) {
        // Linear warmup
        float progress = static_cast<float>(last_epoch_) / static_cast<float>(warmup_steps_);
        float lr = warmup_init_lr_ + (base_lr_ - warmup_init_lr_) * progress;
        set_lr(lr);
    } else {
        // Use underlying scheduler
        scheduler_->step();
    }
}

std::string WarmupLR::to_string() const {
    std::ostringstream oss;
    oss << "WarmupLR(warmup_steps=" << warmup_steps_
        << ", warmup_init_lr=" << warmup_init_lr_
        << ", scheduler=" << scheduler_->to_string() << ")";
    return oss.str();
}

// ============================================================================
// LambdaLR Implementation
// ============================================================================

LambdaLR::LambdaLR(Optimizer& optimizer, LRLambda lr_lambda, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , lr_lambda_(std::move(lr_lambda))
{
}

void LambdaLR::step() {
    last_epoch_++;
    float factor = lr_lambda_(last_epoch_);
    set_lr(base_lr_ * factor);
}

std::string LambdaLR::to_string() const {
    return "LambdaLR(lr_lambda=<function>)";
}

// ============================================================================
// PolynomialLR Implementation
// ============================================================================

PolynomialLR::PolynomialLR(Optimizer& optimizer, int64_t total_iters,
                           float power, float end_lr, int64_t last_epoch)
    : LRScheduler(optimizer, last_epoch)
    , total_iters_(total_iters)
    , power_(power)
    , end_lr_(end_lr)
{
    if (total_iters <= 0) {
        throw std::invalid_argument("total_iters must be positive");
    }
    if (power <= 0.0f) {
        throw std::invalid_argument("power must be positive");
    }
}

void PolynomialLR::step() {
    last_epoch_++;

    float lr;
    if (last_epoch_ >= total_iters_) {
        lr = end_lr_;
    } else {
        float progress = 1.0f - static_cast<float>(last_epoch_) / static_cast<float>(total_iters_);
        lr = (base_lr_ - end_lr_) * std::pow(progress, power_) + end_lr_;
    }

    set_lr(lr);
}

std::string PolynomialLR::to_string() const {
    std::ostringstream oss;
    oss << "PolynomialLR(total_iters=" << total_iters_
        << ", power=" << power_
        << ", end_lr=" << end_lr_ << ")";
    return oss.str();
}

}  // namespace pyflame::optim
