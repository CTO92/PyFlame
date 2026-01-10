#include "pyflame/autograd/grad_mode.hpp"

namespace pyflame::autograd {

// Thread-local storage for gradient mode
thread_local bool GradMode::enabled_ = true;

bool GradMode::is_enabled() {
    return enabled_;
}

void GradMode::set_enabled(bool enabled) {
    enabled_ = enabled;
}

NoGradGuard::NoGradGuard() : prev_mode_(GradMode::is_enabled()) {
    GradMode::set_enabled(false);
}

NoGradGuard::~NoGradGuard() {
    GradMode::set_enabled(prev_mode_);
}

EnableGradGuard::EnableGradGuard() : prev_mode_(GradMode::is_enabled()) {
    GradMode::set_enabled(true);
}

EnableGradGuard::~EnableGradGuard() {
    GradMode::set_enabled(prev_mode_);
}

}  // namespace pyflame::autograd
