// ROCm Memory Management Implementation
// Phase 2: Memory Management

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_memory.hpp"
#include <algorithm>
#include <cmath>
#include <atomic>

namespace pyflame::backend::rocm {

// SECURITY: Maximum allocation size limit (16GB default, can be configured)
constexpr size_t DEFAULT_MAX_ALLOCATION_SIZE = 16ULL * 1024 * 1024 * 1024;

// SECURITY: Atomic counter for pool size to avoid race conditions
namespace {
    std::atomic<size_t> g_total_pooled_size{0};
}

ROCmMemoryManager::ROCmMemoryManager() : config_() {}

ROCmMemoryManager::ROCmMemoryManager(const Config& config) : config_(config) {}

ROCmMemoryManager::~ROCmMemoryManager() {
    clear_pool();
}

size_t ROCmMemoryManager::round_to_bucket(size_t bytes) const {
    // Round up to power of 2 for efficient bucketing
    // Minimum bucket size: 256 bytes
    if (bytes <= 256) return 256;

    // Find next power of 2
    size_t bucket = 256;
    while (bucket < bytes) {
        bucket *= 2;
    }
    return bucket;
}

void* ROCmMemoryManager::try_get_from_pool(size_t bucket_size) {
    if (!config_.enable_pooling) return nullptr;

    auto it = pool_.find(bucket_size);
    if (it != pool_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        // SECURITY FIX (MED-001): Update atomic counter when taking from pool
        g_total_pooled_size.fetch_sub(bucket_size, std::memory_order_relaxed);
        stats_.cache_hits++;
        return ptr;
    }
    stats_.cache_misses++;
    return nullptr;
}

void ROCmMemoryManager::return_to_pool(void* ptr, size_t bucket_size) {
    if (!config_.enable_pooling) {
        HIP_CHECK(hipFree(ptr));
        return;
    }

    // SECURITY FIX (MED-001): Use atomic counter to avoid race condition
    // when checking pool capacity
    size_t max_pool_bytes = config_.max_pool_size_mb * 1024 * 1024;
    size_t current_pooled = g_total_pooled_size.load(std::memory_order_relaxed);

    // Check if adding this would exceed the limit
    if (current_pooled + bucket_size > max_pool_bytes) {
        // Pool full, actually free
        HIP_CHECK(hipFree(ptr));
    } else {
        // Try to atomically increment the pool size
        // If another thread beats us and exceeds the limit, just free
        size_t expected = current_pooled;
        if (g_total_pooled_size.compare_exchange_strong(
                expected, current_pooled + bucket_size,
                std::memory_order_acq_rel)) {
            pool_[bucket_size].push_back(ptr);
        } else {
            // CAS failed, pool may be full now - just free
            HIP_CHECK(hipFree(ptr));
        }
    }
}

void* ROCmMemoryManager::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;

    // SECURITY FIX (HIGH-005): Check allocation size limits
    size_t max_size = config_.max_allocation_size > 0
        ? config_.max_allocation_size
        : DEFAULT_MAX_ALLOCATION_SIZE;

    if (bytes > max_size) {
        throw std::bad_alloc();  // Allocation too large
    }

    std::lock_guard<std::mutex> lock(mutex_);

    size_t bucket_size = round_to_bucket(bytes);

    // Double check bucket size after rounding
    if (bucket_size > max_size) {
        throw std::bad_alloc();
    }

    // Try to get from pool first
    void* ptr = try_get_from_pool(bucket_size);

    if (!ptr) {
        // Allocate new memory
        HIP_CHECK(hipMalloc(&ptr, bucket_size));
    }

    // Track allocation
    allocation_sizes_[ptr] = bucket_size;
    stats_.total_allocated += bucket_size;
    stats_.peak_allocated = std::max(stats_.peak_allocated, stats_.total_allocated);
    stats_.total_allocations++;

    return ptr;
}

void* ROCmMemoryManager::allocate_async(size_t bytes, hipStream_t stream) {
    if (bytes == 0) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);

    size_t bucket_size = round_to_bucket(bytes);

    // Try pool first (even for async, pool lookup is fast)
    void* ptr = try_get_from_pool(bucket_size);

    if (!ptr) {
        // Async allocation (ROCm 5.5+)
        #if HIP_VERSION >= 50500000
        HIP_CHECK(hipMallocAsync(&ptr, bucket_size, stream));
        #else
        // Fallback to sync allocation
        (void)stream;  // Suppress unused parameter warning
        HIP_CHECK(hipMalloc(&ptr, bucket_size));
        #endif
    }

    allocation_sizes_[ptr] = bucket_size;
    stats_.total_allocated += bucket_size;
    stats_.peak_allocated = std::max(stats_.peak_allocated, stats_.total_allocated);
    stats_.total_allocations++;

    return ptr;
}

void ROCmMemoryManager::deallocate(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocation_sizes_.find(ptr);
    if (it == allocation_sizes_.end()) {
        // Unknown pointer - this shouldn't happen
        throw std::runtime_error("ROCmMemoryManager: Attempting to deallocate unknown pointer");
    }

    size_t bucket_size = it->second;
    allocation_sizes_.erase(it);

    stats_.total_allocated -= bucket_size;
    stats_.total_deallocations++;

    // Return to pool instead of freeing
    return_to_pool(ptr, bucket_size);
}

void ROCmMemoryManager::deallocate_async(void* ptr, hipStream_t stream) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocation_sizes_.find(ptr);
    if (it == allocation_sizes_.end()) {
        throw std::runtime_error("ROCmMemoryManager: Attempting to deallocate unknown pointer");
    }

    size_t bucket_size = it->second;
    allocation_sizes_.erase(it);

    stats_.total_allocated -= bucket_size;
    stats_.total_deallocations++;

    if (config_.enable_pooling) {
        // For pooling, we need to sync before returning to pool
        // to ensure the memory is no longer in use
        HIP_CHECK(hipStreamSynchronize(stream));
        return_to_pool(ptr, bucket_size);
    } else {
        #if HIP_VERSION >= 50500000
        HIP_CHECK(hipFreeAsync(ptr, stream));
        #else
        (void)stream;  // Suppress unused parameter warning
        HIP_CHECK(hipFree(ptr));
        #endif
    }
}

void ROCmMemoryManager::copy_host_to_device(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

void ROCmMemoryManager::copy_host_to_device_async(void* dst, const void* src,
                                                   size_t bytes, hipStream_t stream) {
    if (bytes == 0) return;
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream));
}

void ROCmMemoryManager::copy_device_to_host(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}

void ROCmMemoryManager::copy_device_to_host_async(void* dst, const void* src,
                                                   size_t bytes, hipStream_t stream) {
    if (bytes == 0) return;
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream));
}

void ROCmMemoryManager::copy_device_to_device(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
}

void ROCmMemoryManager::copy_device_to_device_async(void* dst, const void* src,
                                                     size_t bytes, hipStream_t stream) {
    if (bytes == 0) return;
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream));
}

void ROCmMemoryManager::memset(void* ptr, int value, size_t bytes) {
    if (bytes == 0) return;
    HIP_CHECK(hipMemset(ptr, value, bytes));
}

void ROCmMemoryManager::memset_async(void* ptr, int value, size_t bytes,
                                      hipStream_t stream) {
    if (bytes == 0) return;
    HIP_CHECK(hipMemsetAsync(ptr, value, bytes, stream));
}

void* ROCmMemoryManager::allocate_pinned(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr;
    HIP_CHECK(hipHostMalloc(&ptr, bytes, hipHostMallocDefault));
    return ptr;
}

void ROCmMemoryManager::deallocate_pinned(void* ptr) {
    if (!ptr) return;
    HIP_CHECK(hipHostFree(ptr));
}

MemoryStats ROCmMemoryManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void ROCmMemoryManager::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = MemoryStats{};
}

void ROCmMemoryManager::clear_pool() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [size, ptrs] : pool_) {
        for (void* ptr : ptrs) {
            hipFree(ptr);  // Don't check error during cleanup
        }
    }
    pool_.clear();

    // SECURITY FIX (MED-001): Reset atomic counter
    g_total_pooled_size.store(0, std::memory_order_relaxed);
}

void ROCmMemoryManager::get_memory_info(size_t* free, size_t* total) const {
    HIP_CHECK(hipMemGetInfo(free, total));
}

// Global singleton
ROCmMemoryManager& get_memory_manager() {
    static ROCmMemoryManager instance;
    return instance;
}

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
