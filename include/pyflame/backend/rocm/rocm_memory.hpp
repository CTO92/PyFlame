#pragma once

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_backend.hpp"
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>

namespace pyflame::backend::rocm {

/// Memory allocation statistics
struct MemoryStats {
    size_t total_allocated = 0;      ///< Total bytes currently allocated
    size_t peak_allocated = 0;       ///< Peak allocation
    size_t total_allocations = 0;    ///< Number of allocations
    size_t total_deallocations = 0;  ///< Number of deallocations
    size_t cache_hits = 0;           ///< Memory pool cache hits
    size_t cache_misses = 0;         ///< Memory pool cache misses
};

/// GPU memory manager with optional pooling
///
/// This class provides efficient GPU memory management with:
/// - Memory pooling to reduce allocation overhead
/// - Host-device and device-device transfers
/// - Pinned host memory for faster transfers
/// - Memory statistics tracking
///
/// Example:
/// @code
/// auto& mem = get_memory_manager();
/// void* gpu_ptr = mem.allocate(1024);
/// mem.copy_host_to_device(gpu_ptr, host_data, 1024);
/// // ... use gpu_ptr ...
/// mem.deallocate(gpu_ptr);
/// @endcode
class ROCmMemoryManager {
public:
    /// Configuration for memory manager
    struct Config {
        bool enable_pooling = true;       ///< Use memory pooling
        size_t pool_size_mb = 256;        ///< Initial pool size
        size_t max_pool_size_mb = 4096;   ///< Maximum pool size
        bool use_pinned_host = true;      ///< Use pinned host memory for transfers
        size_t max_allocation_size = 0;   ///< Max single allocation (0 = default 16GB)

        Config() = default;
    };

    ROCmMemoryManager();  // Uses default Config
    explicit ROCmMemoryManager(const Config& config);
    ~ROCmMemoryManager();

    // Disable copy
    ROCmMemoryManager(const ROCmMemoryManager&) = delete;
    ROCmMemoryManager& operator=(const ROCmMemoryManager&) = delete;

    // ========================================================================
    // GPU Memory Allocation
    // ========================================================================

    /// Allocate GPU memory
    /// @param bytes Number of bytes to allocate
    /// @return Pointer to allocated GPU memory
    /// @throws std::runtime_error on allocation failure
    void* allocate(size_t bytes);

    /// Allocate GPU memory asynchronously (ROCm 5.5+)
    /// @param bytes Number of bytes to allocate
    /// @param stream HIP stream for async allocation
    /// @return Pointer to allocated GPU memory
    void* allocate_async(size_t bytes, hipStream_t stream);

    /// Free GPU memory
    /// @param ptr Pointer to GPU memory (nullptr is safe to pass)
    void deallocate(void* ptr);

    /// Free GPU memory asynchronously
    /// @param ptr Pointer to GPU memory
    /// @param stream HIP stream for async deallocation
    void deallocate_async(void* ptr, hipStream_t stream);

    // ========================================================================
    // Memory Transfers
    // ========================================================================

    /// Copy data from host to device
    /// @param dst Device pointer (destination)
    /// @param src Host pointer (source)
    /// @param bytes Number of bytes to copy
    void copy_host_to_device(void* dst, const void* src, size_t bytes);

    /// Copy data from host to device asynchronously
    /// @param dst Device pointer (destination)
    /// @param src Host pointer (source) - must be pinned for async
    /// @param bytes Number of bytes to copy
    /// @param stream HIP stream for async copy
    void copy_host_to_device_async(void* dst, const void* src, size_t bytes,
                                    hipStream_t stream);

    /// Copy data from device to host
    /// @param dst Host pointer (destination)
    /// @param src Device pointer (source)
    /// @param bytes Number of bytes to copy
    void copy_device_to_host(void* dst, const void* src, size_t bytes);

    /// Copy data from device to host asynchronously
    /// @param dst Host pointer (destination) - must be pinned for async
    /// @param src Device pointer (source)
    /// @param bytes Number of bytes to copy
    /// @param stream HIP stream for async copy
    void copy_device_to_host_async(void* dst, const void* src, size_t bytes,
                                    hipStream_t stream);

    /// Copy data between device buffers
    /// @param dst Device pointer (destination)
    /// @param src Device pointer (source)
    /// @param bytes Number of bytes to copy
    void copy_device_to_device(void* dst, const void* src, size_t bytes);

    /// Copy data between device buffers asynchronously
    /// @param dst Device pointer (destination)
    /// @param src Device pointer (source)
    /// @param bytes Number of bytes to copy
    /// @param stream HIP stream for async copy
    void copy_device_to_device_async(void* dst, const void* src, size_t bytes,
                                      hipStream_t stream);

    // ========================================================================
    // Memory Operations
    // ========================================================================

    /// Set GPU memory to a value
    /// @param ptr Device pointer
    /// @param value Value to set (per byte)
    /// @param bytes Number of bytes to set
    void memset(void* ptr, int value, size_t bytes);

    /// Set GPU memory asynchronously
    /// @param ptr Device pointer
    /// @param value Value to set (per byte)
    /// @param bytes Number of bytes to set
    /// @param stream HIP stream for async memset
    void memset_async(void* ptr, int value, size_t bytes, hipStream_t stream);

    // ========================================================================
    // Pinned Host Memory
    // ========================================================================

    /// Allocate pinned host memory (for faster transfers)
    /// @param bytes Number of bytes to allocate
    /// @return Pointer to pinned host memory
    void* allocate_pinned(size_t bytes);

    /// Free pinned host memory
    /// @param ptr Pointer to pinned host memory
    void deallocate_pinned(void* ptr);

    // ========================================================================
    // Statistics and Management
    // ========================================================================

    /// Get memory statistics
    /// @return Current memory statistics
    MemoryStats get_stats() const;

    /// Reset memory statistics
    void reset_stats();

    /// Clear memory pool (frees cached allocations)
    /// Call this to release pooled memory back to the system
    void clear_pool();

    /// Get GPU memory info
    /// @param free Output: free memory in bytes
    /// @param total Output: total memory in bytes
    void get_memory_info(size_t* free, size_t* total) const;

private:
    Config config_;
    mutable std::mutex mutex_;
    MemoryStats stats_;

    // Memory pool: maps size bucket to list of free pointers
    std::unordered_map<size_t, std::vector<void*>> pool_;

    // Track allocated sizes for deallocation
    std::unordered_map<void*, size_t> allocation_sizes_;

    // Round up to allocation bucket size (power of 2)
    size_t round_to_bucket(size_t bytes) const;

    // Try to get from pool
    void* try_get_from_pool(size_t bucket_size);

    // Return to pool
    void return_to_pool(void* ptr, size_t bucket_size);
};

/// Get the global memory manager instance (thread-safe singleton)
/// @return Reference to the global ROCmMemoryManager
ROCmMemoryManager& get_memory_manager();

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
