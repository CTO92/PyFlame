#pragma once

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_backend.hpp"
#include "pyflame/backend/rocm/rocm_dnn.hpp"
#include <unordered_map>
#include <mutex>
#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <optional>

namespace pyflame::backend::rocm {

// ============================================================================
// MIOpen Algorithm Cache for Auto-tuning
// ============================================================================

/// Key for convolution algorithm cache
struct ConvAlgoCacheKey {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> weight_shape;
    std::vector<int64_t> output_shape;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    int groups;
    DType dtype;

    bool operator==(const ConvAlgoCacheKey& other) const {
        return input_shape == other.input_shape &&
               weight_shape == other.weight_shape &&
               output_shape == other.output_shape &&
               pad_h == other.pad_h && pad_w == other.pad_w &&
               stride_h == other.stride_h && stride_w == other.stride_w &&
               dilation_h == other.dilation_h && dilation_w == other.dilation_w &&
               groups == other.groups && dtype == other.dtype;
    }
};

/// Hash function for ConvAlgoCacheKey
struct ConvAlgoCacheKeyHash {
    size_t operator()(const ConvAlgoCacheKey& key) const {
        size_t hash = 0;
        for (auto d : key.input_shape) hash ^= std::hash<int64_t>{}(d) + 0x9e3779b9;
        for (auto d : key.weight_shape) hash ^= std::hash<int64_t>{}(d) + 0x9e3779b9;
        hash ^= std::hash<int>{}(key.pad_h) + std::hash<int>{}(key.pad_w);
        hash ^= std::hash<int>{}(key.stride_h) + std::hash<int>{}(key.stride_w);
        hash ^= std::hash<int>{}(key.groups);
        hash ^= std::hash<int>{}(static_cast<int>(key.dtype));
        return hash;
    }
};

/// Cached algorithm result
struct CachedAlgoResult {
    miopenConvFwdAlgorithm_t algo;
    size_t workspace_size;
    float time_ms;  // Measured during tuning
};

/// MIOpen auto-tuning cache manager
///
/// Caches the best algorithms found during exhaustive search to avoid
/// repeated tuning overhead. Supports persistence to disk.
///
/// Example:
/// @code
/// MIOpenTuningCache& cache = get_tuning_cache();
/// cache.enable_exhaustive_search(true);  // Enable auto-tuning
/// cache.set_cache_file("/tmp/miopen_cache.bin");
/// @endcode
class MIOpenTuningCache {
public:
    /// Configuration
    struct Config {
        bool enable_exhaustive_search = true;  ///< Use exhaustive search for best algo
        bool enable_caching = true;            ///< Cache found algorithms
        bool enable_persistence = false;       ///< Persist cache to disk
        std::string cache_file;                ///< Path to cache file
        int search_iterations = 100;           ///< Iterations for timing during search

        Config() = default;
    };

    MIOpenTuningCache();
    explicit MIOpenTuningCache(const Config& config);
    ~MIOpenTuningCache();

    /// Enable/disable exhaustive algorithm search
    void enable_exhaustive_search(bool enable) {
        config_.enable_exhaustive_search = enable;
    }

    /// Set cache file path for persistence
    /// @throws std::invalid_argument if path contains traversal sequences or invalid characters
    void set_cache_file(const std::string& path);

    /// Look up cached algorithm
    /// @return Cached result if found, nullptr otherwise
    const CachedAlgoResult* lookup_conv_algo(const ConvAlgoCacheKey& key) const;

    /// Store algorithm result in cache
    void store_conv_algo(const ConvAlgoCacheKey& key, const CachedAlgoResult& result);

    /// Find best convolution algorithm (with auto-tuning)
    ///
    /// If exhaustive search is enabled, this will benchmark multiple algorithms
    /// and cache the best one. Otherwise, it returns a reasonable default.
    ///
    /// @param handle MIOpen handle
    /// @param input_desc Input tensor descriptor
    /// @param input Input data (for benchmarking)
    /// @param weight_desc Weight tensor descriptor
    /// @param weight Weight data (for benchmarking)
    /// @param conv_desc Convolution descriptor
    /// @param output_desc Output tensor descriptor
    /// @param output Output buffer (for benchmarking)
    /// @param workspace Workspace buffer
    /// @param workspace_size Workspace size
    /// @param key Cache key for this configuration
    /// @return Best algorithm and workspace size
    CachedAlgoResult find_best_conv_algo(
        miopenHandle_t handle,
        miopenTensorDescriptor_t input_desc,
        const void* input,
        miopenTensorDescriptor_t weight_desc,
        const void* weight,
        miopenConvolutionDescriptor_t conv_desc,
        miopenTensorDescriptor_t output_desc,
        void* output,
        void* workspace,
        size_t workspace_size,
        const ConvAlgoCacheKey& key
    );

    /// Save cache to disk
    void save_to_disk();

    /// Load cache from disk
    void load_from_disk();

    /// Clear all cached algorithms
    void clear();

    /// Get cache statistics
    struct Stats {
        size_t hits = 0;
        size_t misses = 0;
        size_t entries = 0;
    };
    Stats get_stats() const;

private:
    Config config_;
    mutable std::mutex mutex_;
    std::unordered_map<ConvAlgoCacheKey, CachedAlgoResult, ConvAlgoCacheKeyHash> cache_;
    mutable Stats stats_;
};

/// Get the global tuning cache instance
MIOpenTuningCache& get_tuning_cache();

// ============================================================================
// Pinned Memory Pool
// ============================================================================

/// Pinned memory pool for efficient host-device transfers
///
/// Allocating pinned memory is expensive. This pool maintains a cache of
/// pinned memory buffers for reuse during data transfers.
///
/// Example:
/// @code
/// PinnedMemoryPool& pool = get_pinned_pool();
/// void* pinned = pool.acquire(1024 * 1024);  // 1MB
/// // ... use pinned memory for async transfers ...
/// pool.release(pinned);
/// @endcode
class PinnedMemoryPool {
public:
    struct Config {
        size_t initial_pool_size_mb = 64;   ///< Initial pool size
        size_t max_pool_size_mb = 512;      ///< Maximum pool size
        size_t min_allocation = 4096;       ///< Minimum allocation size

        Config() = default;
    };

    PinnedMemoryPool();
    explicit PinnedMemoryPool(const Config& config);
    ~PinnedMemoryPool();

    // Disable copy
    PinnedMemoryPool(const PinnedMemoryPool&) = delete;
    PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

    /// Acquire pinned memory buffer
    /// @param bytes Minimum size needed
    /// @return Pointer to pinned host memory
    void* acquire(size_t bytes);

    /// Release pinned memory buffer back to pool
    /// @param ptr Pointer previously acquired
    void release(void* ptr);

    /// Pre-warm the pool with specified sizes
    /// @param sizes List of sizes to pre-allocate
    void warmup(const std::vector<size_t>& sizes);

    /// Clear the pool (free all pinned memory)
    void clear();

    /// Get pool statistics
    struct Stats {
        size_t total_allocated = 0;
        size_t pool_size = 0;
        size_t acquisitions = 0;
        size_t releases = 0;
    };
    Stats get_stats() const;

private:
    Config config_;
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<void*>> pool_;
    std::unordered_map<void*, size_t> allocation_sizes_;
    Stats stats_;

    size_t round_to_bucket(size_t bytes) const;
};

/// Get the global pinned memory pool
PinnedMemoryPool& get_pinned_pool();

// ============================================================================
// Kernel Fusion Support
// ============================================================================

/// Fused operation types
enum class FusedOpType {
    MATMUL_BIAS,           ///< matmul + bias add
    MATMUL_BIAS_RELU,      ///< matmul + bias + ReLU
    MATMUL_BIAS_GELU,      ///< matmul + bias + GELU
    CONV_BIAS,             ///< conv + bias add
    CONV_BIAS_RELU,        ///< conv + bias + ReLU
    CONV_BN_RELU,          ///< conv + batch norm + ReLU
    ADD_RELU,              ///< elementwise add + ReLU
    MUL_ADD,               ///< elementwise mul + add (FMA)
};

/// Check if a sequence of operations can be fused
/// @param ops List of operation types
/// @return Fused operation type, or nullopt if not fusable
std::optional<FusedOpType> can_fuse(const std::vector<ir::OpType>& ops);

/// Execute fused matmul + bias + activation
///
/// Computes: output = activation(matmul(A, B) + bias)
///
/// @param fused_type Type of fused operation
/// @param stream HIP stream
/// @param A Input matrix A [M, K]
/// @param B Input matrix B [K, N]
/// @param bias Bias vector [N] (can be nullptr)
/// @param output Output matrix [M, N]
/// @param M Rows of A and output
/// @param K Columns of A, rows of B
/// @param N Columns of B and output
void execute_fused_matmul(
    FusedOpType fused_type,
    hipStream_t stream,
    const float* A,
    const float* B,
    const float* bias,
    float* output,
    int64_t M,
    int64_t K,
    int64_t N
);

/// Execute fused elementwise operations
///
/// @param fused_type Type of fused operation
/// @param stream HIP stream
/// @param a First input
/// @param b Second input (can be nullptr for unary ops)
/// @param c Third input (for ternary ops like FMA)
/// @param output Output buffer
/// @param numel Number of elements
void execute_fused_elementwise(
    FusedOpType fused_type,
    hipStream_t stream,
    const float* a,
    const float* b,
    const float* c,
    float* output,
    int64_t numel
);

// ============================================================================
// Memory Pool Warmup
// ============================================================================

/// Pre-warm the GPU memory pool with common allocation sizes
///
/// This reduces allocation latency during the first forward pass by
/// pre-allocating commonly used buffer sizes.
///
/// @param batch_size Expected batch size
/// @param model_size Estimated model size category (small/medium/large)
void warmup_memory_pool(int batch_size, const std::string& model_size);

/// Memory warmup configuration
struct WarmupConfig {
    std::vector<size_t> gpu_sizes;      ///< GPU allocation sizes to pre-warm
    std::vector<size_t> pinned_sizes;   ///< Pinned memory sizes to pre-warm
    bool warmup_miopen = true;          ///< Warmup MIOpen (runs small operations)
    bool warmup_rocblas = true;         ///< Warmup rocBLAS
};

/// Advanced warmup with custom configuration
void warmup_pools(const WarmupConfig& config);

// ============================================================================
// Performance Profiling
// ============================================================================

/// Simple kernel timer
class KernelTimer {
public:
    KernelTimer(hipStream_t stream, bool enabled = true);
    ~KernelTimer();

    /// Start timing
    void start();

    /// Stop timing and return elapsed milliseconds
    float stop();

    /// Get last recorded time
    float elapsed_ms() const { return elapsed_ms_; }

private:
    hipStream_t stream_;
    hipEvent_t start_event_;
    hipEvent_t stop_event_;
    bool enabled_;
    float elapsed_ms_ = 0.0f;
};

/// RAII scope timer
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, hipStream_t stream);
    ~ScopedTimer();

private:
    std::string name_;
    KernelTimer timer_;
};

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
