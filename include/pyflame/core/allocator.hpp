#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>

namespace pyflame {

// ============================================================================
// Configurable alignment constants
// ============================================================================

/// Default cache line size - can be overridden at compile time
/// Common values: 64 (x86/x64), 128 (some ARM), 32 (older systems)
#ifndef PYFLAME_CACHE_LINE_SIZE
#define PYFLAME_CACHE_LINE_SIZE 64
#endif

/// Maximum allowed alignment (must be power of 2)
#ifndef PYFLAME_MAX_ALIGNMENT
#define PYFLAME_MAX_ALIGNMENT 4096
#endif

/// Minimum allowed alignment (must be power of 2)
#ifndef PYFLAME_MIN_ALIGNMENT
#define PYFLAME_MIN_ALIGNMENT 1
#endif

// Compile-time validation
static_assert((PYFLAME_CACHE_LINE_SIZE & (PYFLAME_CACHE_LINE_SIZE - 1)) == 0,
              "PYFLAME_CACHE_LINE_SIZE must be a power of 2");
static_assert((PYFLAME_MAX_ALIGNMENT & (PYFLAME_MAX_ALIGNMENT - 1)) == 0,
              "PYFLAME_MAX_ALIGNMENT must be a power of 2");
static_assert(PYFLAME_MIN_ALIGNMENT >= 1 && PYFLAME_MIN_ALIGNMENT <= PYFLAME_MAX_ALIGNMENT,
              "PYFLAME_MIN_ALIGNMENT must be between 1 and PYFLAME_MAX_ALIGNMENT");

/// Aligned memory allocator for efficient data access
class Allocator {
public:
    /// Default alignment based on cache line size
    static constexpr size_t DEFAULT_ALIGNMENT = PYFLAME_CACHE_LINE_SIZE;

    /// Maximum alignment value
    static constexpr size_t MAX_ALIGNMENT = PYFLAME_MAX_ALIGNMENT;

    /// Minimum alignment value
    static constexpr size_t MIN_ALIGNMENT = PYFLAME_MIN_ALIGNMENT;

    /// Check if alignment is valid (power of 2 and within bounds)
    static constexpr bool is_valid_alignment(size_t alignment) {
        return alignment >= MIN_ALIGNMENT &&
               alignment <= MAX_ALIGNMENT &&
               (alignment & (alignment - 1)) == 0;  // Power of 2 check
    }

    /// Allocate aligned memory
    /// @param bytes Number of bytes to allocate
    /// @param alignment Alignment requirement (must be power of 2, >= MIN_ALIGNMENT, <= MAX_ALIGNMENT)
    /// @throws std::invalid_argument if alignment is invalid
    /// @throws std::bad_alloc if allocation fails
    static void* allocate(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT) {
        if (bytes == 0) return nullptr;

        // Validate alignment at runtime
        if (!is_valid_alignment(alignment)) {
            throw std::invalid_argument(
                "Invalid alignment value: " + std::to_string(alignment) +
                " (must be power of 2, >= " + std::to_string(MIN_ALIGNMENT) +
                ", <= " + std::to_string(MAX_ALIGNMENT) + ")");
        }

#if defined(_MSC_VER)
        void* ptr = _aligned_malloc(bytes, alignment);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, bytes) != 0) {
            ptr = nullptr;
        }
#endif
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    /// Deallocate memory
    static void deallocate(void* ptr) {
        if (!ptr) return;
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    /// Allocate and zero-initialize memory
    /// @param bytes Number of bytes to allocate
    /// @param alignment Alignment requirement
    /// @throws std::invalid_argument if alignment is invalid
    /// @throws std::bad_alloc if allocation fails
    static void* allocate_zeroed(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT) {
        void* ptr = allocate(bytes, alignment);
        if (ptr) {
            std::memset(ptr, 0, bytes);
        }
        return ptr;
    }

    /// Custom deleter for unique_ptr
    struct Deleter {
        void operator()(void* ptr) const {
            deallocate(ptr);
        }
    };

    /// Typed deleter for unique_ptr
    template<typename T>
    struct TypedDeleter {
        void operator()(T* ptr) const {
            deallocate(ptr);
        }
    };
};

/// Unique pointer with aligned allocator
template<typename T>
using AlignedUniquePtr = std::unique_ptr<T[], Allocator::TypedDeleter<T>>;

/// Create an aligned unique pointer for an array
template<typename T>
AlignedUniquePtr<T> make_aligned_unique(size_t count, size_t alignment = Allocator::DEFAULT_ALIGNMENT) {
    T* ptr = static_cast<T*>(Allocator::allocate(count * sizeof(T), alignment));
    return AlignedUniquePtr<T>(ptr);
}

/// Shared pointer with aligned allocator
template<typename T>
std::shared_ptr<T> make_aligned_shared(size_t count, size_t alignment = Allocator::DEFAULT_ALIGNMENT) {
    T* ptr = static_cast<T*>(Allocator::allocate(count * sizeof(T), alignment));
    return std::shared_ptr<T>(ptr, [](T* p) { Allocator::deallocate(p); });
}

}  // namespace pyflame
