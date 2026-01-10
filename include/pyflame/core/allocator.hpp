#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace pyflame {

/// Aligned memory allocator for efficient data access
class Allocator {
public:
    static constexpr size_t DEFAULT_ALIGNMENT = 64;  // Cache line size

    /// Allocate aligned memory
    static void* allocate(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT) {
        if (bytes == 0) return nullptr;

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
