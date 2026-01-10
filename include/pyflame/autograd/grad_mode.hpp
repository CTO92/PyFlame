#pragma once

namespace pyflame::autograd {

/// Global gradient computation state
/// Controls whether operations track gradients
class GradMode {
public:
    /// Check if gradient computation is enabled
    static bool is_enabled();

    /// Enable or disable gradient computation
    static void set_enabled(bool enabled);

private:
    static thread_local bool enabled_;
};

/// RAII class for temporarily disabling gradients
/// Usage:
///   {
///     NoGradGuard guard;
///     // gradients disabled in this scope
///   }
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    // Non-copyable
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_mode_;
};

/// RAII class for temporarily enabling gradients
/// Usage:
///   {
///     EnableGradGuard guard;
///     // gradients enabled in this scope
///   }
class EnableGradGuard {
public:
    EnableGradGuard();
    ~EnableGradGuard();

    // Non-copyable
    EnableGradGuard(const EnableGradGuard&) = delete;
    EnableGradGuard& operator=(const EnableGradGuard&) = delete;

private:
    bool prev_mode_;
};

}  // namespace pyflame::autograd
