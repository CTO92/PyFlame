#pragma once

#include <cstdint>
#include <stdexcept>
#include <limits>
#include <type_traits>

namespace pyflame {

/// Safe integer multiplication with overflow detection
/// Returns true if overflow occurred, false otherwise
template<typename T>
inline bool safe_mul_overflow(T a, T b, T* result) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");

#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, result);
#elif defined(_MSC_VER)
    // MSVC doesn't have __builtin_mul_overflow
    if constexpr (std::is_signed_v<T>) {
        // Check for signed overflow
        if (a > 0 && b > 0 && a > std::numeric_limits<T>::max() / b) {
            return true;
        }
        if (a > 0 && b < 0 && b < std::numeric_limits<T>::min() / a) {
            return true;
        }
        if (a < 0 && b > 0 && a < std::numeric_limits<T>::min() / b) {
            return true;
        }
        if (a < 0 && b < 0 && a != 0 && b < std::numeric_limits<T>::max() / a) {
            return true;
        }
    } else {
        // Unsigned overflow check
        if (b != 0 && a > std::numeric_limits<T>::max() / b) {
            return true;
        }
    }
    *result = a * b;
    return false;
#else
    // Generic fallback
    if constexpr (std::is_signed_v<T>) {
        if (a > 0 && b > 0 && a > std::numeric_limits<T>::max() / b) {
            return true;
        }
        if (a > 0 && b < 0 && b < std::numeric_limits<T>::min() / a) {
            return true;
        }
        if (a < 0 && b > 0 && a < std::numeric_limits<T>::min() / b) {
            return true;
        }
        if (a < 0 && b < 0 && a != 0 && b < std::numeric_limits<T>::max() / a) {
            return true;
        }
    } else {
        if (b != 0 && a > std::numeric_limits<T>::max() / b) {
            return true;
        }
    }
    *result = a * b;
    return false;
#endif
}

/// Safe integer addition with overflow detection
template<typename T>
inline bool safe_add_overflow(T a, T b, T* result) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");

#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, result);
#else
    if constexpr (std::is_signed_v<T>) {
        if (b > 0 && a > std::numeric_limits<T>::max() - b) {
            return true;
        }
        if (b < 0 && a < std::numeric_limits<T>::min() - b) {
            return true;
        }
    } else {
        if (a > std::numeric_limits<T>::max() - b) {
            return true;
        }
    }
    *result = a + b;
    return false;
#endif
}

/// Checked multiplication that throws on overflow
template<typename T>
inline T checked_mul(T a, T b) {
    T result;
    if (safe_mul_overflow(a, b, &result)) {
        throw std::overflow_error("Integer overflow in multiplication");
    }
    return result;
}

/// Checked addition that throws on overflow
template<typename T>
inline T checked_add(T a, T b) {
    T result;
    if (safe_add_overflow(a, b, &result)) {
        throw std::overflow_error("Integer overflow in addition");
    }
    return result;
}

/// Safe computation of total elements from shape with overflow checking
inline int64_t safe_numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;  // Scalar

    int64_t n = 1;
    for (auto d : shape) {
        if (d < 0) {
            throw std::invalid_argument("Shape dimension cannot be negative");
        }
        if (safe_mul_overflow(n, d, &n)) {
            throw std::overflow_error("Shape overflow: total elements exceed int64_t maximum");
        }
    }
    return n;
}

/// Safe computation of byte size with overflow checking
inline size_t safe_size_bytes(int64_t numel, size_t dtype_size) {
    if (numel < 0) {
        throw std::invalid_argument("Number of elements cannot be negative");
    }

    // Check if numel * dtype_size would overflow size_t
    size_t numel_sz = static_cast<size_t>(numel);
    size_t result;
    if (safe_mul_overflow(numel_sz, dtype_size, &result)) {
        throw std::overflow_error("Size overflow: byte size exceeds size_t maximum");
    }
    return result;
}

/// Maximum number of dimensions supported
constexpr int MAX_TENSOR_DIMS = 16;

/// Maximum number of elements (prevent pathological allocations)
constexpr int64_t MAX_TENSOR_NUMEL = INT64_C(1) << 40;  // ~1 trillion elements

/// Validate shape dimensions
inline void validate_shape(const std::vector<int64_t>& shape) {
    if (shape.size() > MAX_TENSOR_DIMS) {
        throw std::invalid_argument(
            "Shape has too many dimensions: " + std::to_string(shape.size()) +
            " > " + std::to_string(MAX_TENSOR_DIMS));
    }

    int64_t numel = safe_numel(shape);
    if (numel > MAX_TENSOR_NUMEL) {
        throw std::overflow_error(
            "Shape would create too many elements: " + std::to_string(numel) +
            " > " + std::to_string(MAX_TENSOR_NUMEL));
    }
}

}  // namespace pyflame
