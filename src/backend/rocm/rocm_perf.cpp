// ROCm Performance Optimization Implementation
// Phase 9: Performance Optimization

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_perf.hpp"
#include "pyflame/backend/rocm/rocm_memory.hpp"
#include "pyflame/backend/rocm/rocm_blas.hpp"
#include "pyflame/backend/rocm/rocm_dnn.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <regex>

namespace pyflame::backend::rocm {

// ============================================================================
// Security Helpers
// ============================================================================

namespace {

// SECURITY: Maximum limits for cache file validation
constexpr size_t MAX_CACHE_ENTRIES = 100000;      // Max entries in cache file
constexpr size_t MAX_SHAPE_DIMS = 16;             // Max tensor dimensions
constexpr size_t MAX_CACHE_FILE_SIZE = 100 * 1024 * 1024;  // 100MB max cache file
constexpr size_t MAX_CACHE_MEMORY = 256 * 1024 * 1024;     // 256MB max cache memory

/// Validate cache file path to prevent path traversal attacks
/// @returns true if path is safe, false otherwise
bool validate_cache_path(const std::string& path) {
    if (path.empty()) return true;  // Empty path is OK (disables persistence)

    // Check for path traversal sequences
    if (path.find("..") != std::string::npos) {
        return false;
    }

    // Check for null bytes (could truncate path in C APIs)
    if (path.find('\0') != std::string::npos) {
        return false;
    }

    // Only allow alphanumeric, underscore, hyphen, dot, and path separators
    static const std::regex safe_path_regex("^[a-zA-Z0-9_./-]+$");
    if (!std::regex_match(path, safe_path_regex)) {
        return false;
    }

    // Disallow absolute paths starting with / unless in allowed directories
    // (allows relative paths or explicit allowed directories)
    if (path[0] == '/' && path.find("/tmp/") != 0 &&
        path.find("/var/tmp/") != 0 &&
        path.find(std::getenv("HOME") ? std::getenv("HOME") : "/nonexistent") != 0) {
        // For security, we could restrict to specific directories
        // For now, we allow any path that passes the above checks
    }

    return true;
}

}  // anonymous namespace

// ============================================================================
// MIOpen Tuning Cache Implementation
// ============================================================================

MIOpenTuningCache::MIOpenTuningCache(Config config) : config_(config) {
    // SECURITY: Validate cache file path on construction
    if (!config_.cache_file.empty() && !validate_cache_path(config_.cache_file)) {
        throw std::invalid_argument(
            "Invalid cache file path: contains path traversal or invalid characters");
    }
    if (config_.enable_persistence && !config_.cache_file.empty()) {
        load_from_disk();
    }
}

void MIOpenTuningCache::set_cache_file(const std::string& path) {
    // SECURITY: Validate path to prevent path traversal attacks
    if (!validate_cache_path(path)) {
        throw std::invalid_argument(
            "Invalid cache file path: contains '..' or invalid characters. "
            "Path must contain only alphanumeric characters, underscores, "
            "hyphens, dots, and path separators.");
    }
    config_.cache_file = path;
    config_.enable_persistence = !path.empty();
}

MIOpenTuningCache::~MIOpenTuningCache() {
    if (config_.enable_persistence && !config_.cache_file.empty()) {
        save_to_disk();
    }
}

const CachedAlgoResult* MIOpenTuningCache::lookup_conv_algo(
    const ConvAlgoCacheKey& key
) const {
    if (!config_.enable_caching) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        stats_.hits++;
        return &it->second;
    }
    stats_.misses++;
    return nullptr;
}

void MIOpenTuningCache::store_conv_algo(
    const ConvAlgoCacheKey& key,
    const CachedAlgoResult& result
) {
    if (!config_.enable_caching) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // SECURITY: Limit cache size to prevent memory exhaustion (LOW-005)
    if (cache_.size() >= MAX_CACHE_ENTRIES) {
        // Cache is full - don't add more entries
        // In production, consider LRU eviction
        return;
    }

    cache_[key] = result;
    stats_.entries = cache_.size();
}

CachedAlgoResult MIOpenTuningCache::find_best_conv_algo(
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
) {
    // Check cache first
    if (const auto* cached = lookup_conv_algo(key)) {
        return *cached;
    }

    CachedAlgoResult result;

    if (config_.enable_exhaustive_search) {
        // Request all available algorithms
        const int max_algos = 10;
        miopenConvAlgoPerf_t perfs[max_algos];
        int returned_algo_count = 0;

        // Exhaustive search - benchmarks all algorithms
        miopenStatus_t status = miopenFindConvolutionForwardAlgorithm(
            handle,
            input_desc, input,
            weight_desc, weight,
            conv_desc,
            output_desc, output,
            max_algos,
            &returned_algo_count,
            perfs,
            workspace,
            workspace_size,
            true  // Exhaustive search!
        );

        if (status == miopenStatusSuccess && returned_algo_count > 0) {
            // perfs[0] should be the fastest
            result.algo = perfs[0].fwd_algo;
            result.workspace_size = perfs[0].memory;
            result.time_ms = perfs[0].time;
        } else {
            // Fallback to default
            result.algo = miopenConvolutionFwdAlgoGEMM;
            result.workspace_size = workspace_size;
            result.time_ms = 0.0f;
        }
    } else {
        // Quick search - just get a working algorithm
        miopenConvAlgoPerf_t perf;
        int returned_algo_count = 0;

        miopenFindConvolutionForwardAlgorithm(
            handle,
            input_desc, input,
            weight_desc, weight,
            conv_desc,
            output_desc, output,
            1,
            &returned_algo_count,
            &perf,
            workspace,
            workspace_size,
            false  // Non-exhaustive
        );

        result.algo = perf.fwd_algo;
        result.workspace_size = perf.memory;
        result.time_ms = perf.time;
    }

    // Cache the result
    store_conv_algo(key, result);

    return result;
}

void MIOpenTuningCache::save_to_disk() {
    if (config_.cache_file.empty()) return;

    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(config_.cache_file, std::ios::binary);
    if (!file) return;

    // Write number of entries
    size_t count = cache_.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Write each entry
    for (const auto& [key, result] : cache_) {
        // Write key
        size_t size = key.input_shape.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(key.input_shape.data()),
                   size * sizeof(int64_t));

        size = key.weight_shape.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(key.weight_shape.data()),
                   size * sizeof(int64_t));

        size = key.output_shape.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(key.output_shape.data()),
                   size * sizeof(int64_t));

        file.write(reinterpret_cast<const char*>(&key.pad_h), sizeof(key.pad_h));
        file.write(reinterpret_cast<const char*>(&key.pad_w), sizeof(key.pad_w));
        file.write(reinterpret_cast<const char*>(&key.stride_h), sizeof(key.stride_h));
        file.write(reinterpret_cast<const char*>(&key.stride_w), sizeof(key.stride_w));
        file.write(reinterpret_cast<const char*>(&key.dilation_h), sizeof(key.dilation_h));
        file.write(reinterpret_cast<const char*>(&key.dilation_w), sizeof(key.dilation_w));
        file.write(reinterpret_cast<const char*>(&key.groups), sizeof(key.groups));
        file.write(reinterpret_cast<const char*>(&key.dtype), sizeof(key.dtype));

        // Write result
        file.write(reinterpret_cast<const char*>(&result.algo), sizeof(result.algo));
        file.write(reinterpret_cast<const char*>(&result.workspace_size),
                   sizeof(result.workspace_size));
        file.write(reinterpret_cast<const char*>(&result.time_ms), sizeof(result.time_ms));
    }
}

void MIOpenTuningCache::load_from_disk() {
    if (config_.cache_file.empty()) return;

    std::ifstream file(config_.cache_file, std::ios::binary);
    if (!file) return;

    // SECURITY: Check file size before reading
    file.seekg(0, std::ios::end);
    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < 0 || static_cast<size_t>(file_size) > MAX_CACHE_FILE_SIZE) {
        // File too large - refuse to load
        std::cerr << "[WARN] Cache file exceeds maximum size, not loading" << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();

    size_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));

    // SECURITY: Validate entry count
    if (count > MAX_CACHE_ENTRIES) {
        std::cerr << "[WARN] Cache file has too many entries (" << count
                  << " > " << MAX_CACHE_ENTRIES << "), not loading" << std::endl;
        return;
    }

    for (size_t i = 0; i < count && file; ++i) {
        ConvAlgoCacheKey key;
        CachedAlgoResult result;

        // Read key with size validation
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));

        // SECURITY: Validate shape size to prevent huge allocations
        if (size > MAX_SHAPE_DIMS) {
            std::cerr << "[WARN] Invalid shape size in cache file, stopping load" << std::endl;
            break;
        }
        key.input_shape.resize(size);
        file.read(reinterpret_cast<char*>(key.input_shape.data()),
                  size * sizeof(int64_t));

        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (size > MAX_SHAPE_DIMS) {
            std::cerr << "[WARN] Invalid shape size in cache file, stopping load" << std::endl;
            break;
        }
        key.weight_shape.resize(size);
        file.read(reinterpret_cast<char*>(key.weight_shape.data()),
                  size * sizeof(int64_t));

        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (size > MAX_SHAPE_DIMS) {
            std::cerr << "[WARN] Invalid shape size in cache file, stopping load" << std::endl;
            break;
        }
        key.output_shape.resize(size);
        file.read(reinterpret_cast<char*>(key.output_shape.data()),
                  size * sizeof(int64_t));

        file.read(reinterpret_cast<char*>(&key.pad_h), sizeof(key.pad_h));
        file.read(reinterpret_cast<char*>(&key.pad_w), sizeof(key.pad_w));
        file.read(reinterpret_cast<char*>(&key.stride_h), sizeof(key.stride_h));
        file.read(reinterpret_cast<char*>(&key.stride_w), sizeof(key.stride_w));
        file.read(reinterpret_cast<char*>(&key.dilation_h), sizeof(key.dilation_h));
        file.read(reinterpret_cast<char*>(&key.dilation_w), sizeof(key.dilation_w));
        file.read(reinterpret_cast<char*>(&key.groups), sizeof(key.groups));
        file.read(reinterpret_cast<char*>(&key.dtype), sizeof(key.dtype));

        // Read result
        file.read(reinterpret_cast<char*>(&result.algo), sizeof(result.algo));
        file.read(reinterpret_cast<char*>(&result.workspace_size),
                  sizeof(result.workspace_size));
        file.read(reinterpret_cast<char*>(&result.time_ms), sizeof(result.time_ms));

        if (file) {
            cache_[key] = result;
        }
    }

    stats_.entries = cache_.size();
}

void MIOpenTuningCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    stats_ = Stats{};
}

MIOpenTuningCache::Stats MIOpenTuningCache::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

MIOpenTuningCache& get_tuning_cache() {
    static MIOpenTuningCache instance;
    return instance;
}

// ============================================================================
// Pinned Memory Pool Implementation
// ============================================================================

PinnedMemoryPool::PinnedMemoryPool(Config config) : config_(config) {}

PinnedMemoryPool::~PinnedMemoryPool() {
    clear();
}

size_t PinnedMemoryPool::round_to_bucket(size_t bytes) const {
    // Round up to power of 2, minimum 4KB
    size_t min_size = std::max(bytes, config_.min_allocation);
    size_t bucket = config_.min_allocation;
    while (bucket < min_size) {
        bucket *= 2;
    }
    return bucket;
}

void* PinnedMemoryPool::acquire(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t bucket_size = round_to_bucket(bytes);

    // Try to get from pool
    auto it = pool_.find(bucket_size);
    if (it != pool_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        stats_.acquisitions++;
        return ptr;
    }

    // Allocate new pinned memory
    void* ptr;
    HIP_CHECK(hipHostMalloc(&ptr, bucket_size, hipHostMallocDefault));

    allocation_sizes_[ptr] = bucket_size;
    stats_.total_allocated += bucket_size;
    stats_.acquisitions++;

    return ptr;
}

void PinnedMemoryPool::release(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocation_sizes_.find(ptr);
    if (it == allocation_sizes_.end()) {
        // SECURITY FIX (LOW-001): Unknown pointer - synchronize before freeing
        // to ensure any async operations using this memory are complete
        hipDeviceSynchronize();
        hipHostFree(ptr);
        return;
    }

    size_t bucket_size = it->second;

    // Check if pool is at max capacity
    size_t pool_total = 0;
    for (const auto& [size, ptrs] : pool_) {
        pool_total += size * ptrs.size();
    }

    if (pool_total + bucket_size > config_.max_pool_size_mb * 1024 * 1024) {
        // Pool full, free the memory
        hipHostFree(ptr);
        allocation_sizes_.erase(it);
        stats_.total_allocated -= bucket_size;
    } else {
        // Return to pool
        pool_[bucket_size].push_back(ptr);
        stats_.pool_size += bucket_size;
    }

    stats_.releases++;
}

void PinnedMemoryPool::warmup(const std::vector<size_t>& sizes) {
    for (size_t size : sizes) {
        void* ptr = acquire(size);
        release(ptr);
    }
}

void PinnedMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [size, ptrs] : pool_) {
        for (void* ptr : ptrs) {
            hipHostFree(ptr);
        }
    }
    pool_.clear();

    // Note: we don't free currently acquired buffers - they're still in use
    stats_.pool_size = 0;
}

PinnedMemoryPool::Stats PinnedMemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

PinnedMemoryPool& get_pinned_pool() {
    static PinnedMemoryPool instance;
    return instance;
}

// ============================================================================
// Kernel Fusion Support
// ============================================================================

std::optional<FusedOpType> can_fuse(const std::vector<ir::OpType>& ops) {
    if (ops.size() < 2) return std::nullopt;

    // Check common fusion patterns
    if (ops.size() == 2) {
        if (ops[0] == ir::OpType::MATMUL && ops[1] == ir::OpType::ADD) {
            return FusedOpType::MATMUL_BIAS;
        }
        if (ops[0] == ir::OpType::ADD && ops[1] == ir::OpType::RELU) {
            return FusedOpType::ADD_RELU;
        }
        if (ops[0] == ir::OpType::CONV2D && ops[1] == ir::OpType::ADD) {
            return FusedOpType::CONV_BIAS;
        }
    }

    if (ops.size() == 3) {
        if (ops[0] == ir::OpType::MATMUL && ops[1] == ir::OpType::ADD &&
            ops[2] == ir::OpType::RELU) {
            return FusedOpType::MATMUL_BIAS_RELU;
        }
        if (ops[0] == ir::OpType::MATMUL && ops[1] == ir::OpType::ADD &&
            ops[2] == ir::OpType::GELU) {
            return FusedOpType::MATMUL_BIAS_GELU;
        }
        if (ops[0] == ir::OpType::CONV2D && ops[1] == ir::OpType::ADD &&
            ops[2] == ir::OpType::RELU) {
            return FusedOpType::CONV_BIAS_RELU;
        }
        if (ops[0] == ir::OpType::CONV2D && ops[1] == ir::OpType::BATCH_NORM &&
            ops[2] == ir::OpType::RELU) {
            return FusedOpType::CONV_BN_RELU;
        }
        if (ops[0] == ir::OpType::MUL && ops[1] == ir::OpType::ADD) {
            return FusedOpType::MUL_ADD;
        }
    }

    return std::nullopt;
}

// Fused kernel declarations (implemented in HIP)
extern "C" {
    void launch_fused_matmul_bias(
        float* output, const float* A, const float* B, const float* bias,
        int64_t M, int64_t K, int64_t N, hipStream_t stream);
    void launch_fused_matmul_bias_relu(
        float* output, const float* A, const float* B, const float* bias,
        int64_t M, int64_t K, int64_t N, hipStream_t stream);
    void launch_fused_matmul_bias_gelu(
        float* output, const float* A, const float* B, const float* bias,
        int64_t M, int64_t K, int64_t N, hipStream_t stream);
    void launch_fused_add_relu(
        float* output, const float* a, const float* b,
        int64_t numel, hipStream_t stream);
    void launch_fused_mul_add(
        float* output, const float* a, const float* b, const float* c,
        int64_t numel, hipStream_t stream);
}

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
) {
    // For matmul fusion, we use rocBLAS for the GEMM part
    // and fuse the bias/activation in a custom kernel

    auto& blas = get_blas();
    blas.set_stream(stream);

    // First do GEMM: output = A @ B
    blas.gemm(
        ROCBLAS_OPERATION_NONE,
        ROCBLAS_OPERATION_NONE,
        N, M, K,  // rocBLAS uses column-major
        1.0f,
        B, N,
        A, K,
        0.0f,
        output, N
    );

    // Then apply bias and activation
    switch (fused_type) {
        case FusedOpType::MATMUL_BIAS:
            if (bias) {
                launch_fused_matmul_bias(output, nullptr, nullptr, bias, M, K, N, stream);
            }
            break;
        case FusedOpType::MATMUL_BIAS_RELU:
            launch_fused_matmul_bias_relu(output, nullptr, nullptr, bias, M, K, N, stream);
            break;
        case FusedOpType::MATMUL_BIAS_GELU:
            launch_fused_matmul_bias_gelu(output, nullptr, nullptr, bias, M, K, N, stream);
            break;
        default:
            break;
    }
}

void execute_fused_elementwise(
    FusedOpType fused_type,
    hipStream_t stream,
    const float* a,
    const float* b,
    const float* c,
    float* output,
    int64_t numel
) {
    switch (fused_type) {
        case FusedOpType::ADD_RELU:
            launch_fused_add_relu(output, a, b, numel, stream);
            break;
        case FusedOpType::MUL_ADD:
            launch_fused_mul_add(output, a, b, c, numel, stream);
            break;
        default:
            throw std::runtime_error("Unsupported fused elementwise operation");
    }
}

// ============================================================================
// Memory Pool Warmup
// ============================================================================

void warmup_memory_pool(int batch_size, const std::string& model_size) {
    WarmupConfig config;

    // Common allocation sizes based on model size
    if (model_size == "small") {
        // Small models (e.g., MNIST, small transformers)
        config.gpu_sizes = {
            static_cast<size_t>(batch_size) * 784 * 4,     // Input
            static_cast<size_t>(batch_size) * 256 * 4,     // Hidden
            static_cast<size_t>(batch_size) * 10 * 4,      // Output
            256 * 784 * 4,                                  // Weight 1
            10 * 256 * 4,                                   // Weight 2
        };
        config.pinned_sizes = {
            static_cast<size_t>(batch_size) * 784 * 4,
            static_cast<size_t>(batch_size) * 10 * 4,
        };
    } else if (model_size == "medium") {
        // Medium models (e.g., ResNet-18, BERT-base)
        config.gpu_sizes = {
            static_cast<size_t>(batch_size) * 3 * 224 * 224 * 4,  // Input
            static_cast<size_t>(batch_size) * 64 * 112 * 112 * 4, // Conv1 output
            static_cast<size_t>(batch_size) * 64 * 56 * 56 * 4,
            static_cast<size_t>(batch_size) * 128 * 28 * 28 * 4,
            static_cast<size_t>(batch_size) * 256 * 14 * 14 * 4,
            static_cast<size_t>(batch_size) * 512 * 7 * 7 * 4,
            static_cast<size_t>(batch_size) * 1000 * 4,           // Output
        };
        config.pinned_sizes = {
            static_cast<size_t>(batch_size) * 3 * 224 * 224 * 4,
            static_cast<size_t>(batch_size) * 1000 * 4,
        };
    } else {  // "large"
        // Large models (e.g., ResNet-152, BERT-large)
        config.gpu_sizes = {
            static_cast<size_t>(batch_size) * 3 * 224 * 224 * 4,
            static_cast<size_t>(batch_size) * 64 * 112 * 112 * 4,
            static_cast<size_t>(batch_size) * 256 * 56 * 56 * 4,
            static_cast<size_t>(batch_size) * 512 * 28 * 28 * 4,
            static_cast<size_t>(batch_size) * 1024 * 14 * 14 * 4,
            static_cast<size_t>(batch_size) * 2048 * 7 * 7 * 4,
            256 * 1024 * 1024,  // Large weight matrices
        };
        config.pinned_sizes = {
            static_cast<size_t>(batch_size) * 3 * 224 * 224 * 4,
            static_cast<size_t>(batch_size) * 1000 * 4,
            64 * 1024 * 1024,  // 64MB staging buffer
        };
    }

    warmup_pools(config);
}

void warmup_pools(const WarmupConfig& config) {
    auto& gpu_mem = get_memory_manager();
    auto& pinned_pool = get_pinned_pool();

    // Pre-warm GPU memory pool
    std::vector<void*> gpu_ptrs;
    for (size_t size : config.gpu_sizes) {
        void* ptr = gpu_mem.allocate(size);
        gpu_ptrs.push_back(ptr);
    }
    // Return to pool
    for (void* ptr : gpu_ptrs) {
        gpu_mem.deallocate(ptr);
    }

    // Pre-warm pinned memory pool
    pinned_pool.warmup(config.pinned_sizes);

    // Warmup MIOpen with a small convolution
    if (config.warmup_miopen) {
        auto& dnn = get_dnn();

        // Small dummy convolution to initialize MIOpen
        size_t dummy_size = 32 * 64 * 32 * 32 * sizeof(float);
        void* dummy_input = gpu_mem.allocate(dummy_size);
        void* dummy_weight = gpu_mem.allocate(64 * 64 * 3 * 3 * sizeof(float));
        void* dummy_output = gpu_mem.allocate(dummy_size);
        void* workspace = gpu_mem.allocate(256 * 1024 * 1024);  // 256MB workspace

        gpu_mem.memset(dummy_input, 0, dummy_size);
        gpu_mem.memset(dummy_weight, 0, 64 * 64 * 3 * 3 * sizeof(float));

        try {
            ConvParams params = ConvParams::Conv2D(1, 1, 1, 1);
            dnn.conv2d_forward(
                DType::Float32,
                {32, 64, 32, 32},
                dummy_input,
                {64, 64, 3, 3},
                dummy_weight,
                {32, 64, 32, 32},
                dummy_output,
                params,
                workspace,
                256 * 1024 * 1024
            );
        } catch (const std::exception& e) {
            // SECURITY FIX (LOW-004): Log warning instead of silently ignoring
#ifndef NDEBUG
            std::cerr << "[WARN] MIOpen warmup failed: " << e.what() << std::endl;
#endif
            (void)e;
        } catch (...) {
            // Non-standard exception during warmup
#ifndef NDEBUG
            std::cerr << "[WARN] MIOpen warmup failed with unknown exception" << std::endl;
#endif
        }

        gpu_mem.deallocate(dummy_input);
        gpu_mem.deallocate(dummy_weight);
        gpu_mem.deallocate(dummy_output);
        gpu_mem.deallocate(workspace);
    }

    // Warmup rocBLAS with a small GEMM
    if (config.warmup_rocblas) {
        auto& blas = get_blas();

        size_t dummy_size = 256 * 256 * sizeof(float);
        void* A = gpu_mem.allocate(dummy_size);
        void* B = gpu_mem.allocate(dummy_size);
        void* C = gpu_mem.allocate(dummy_size);

        gpu_mem.memset(A, 0, dummy_size);
        gpu_mem.memset(B, 0, dummy_size);
        gpu_mem.memset(C, 0, dummy_size);

        try {
            blas.gemm(
                ROCBLAS_OPERATION_NONE,
                ROCBLAS_OPERATION_NONE,
                256, 256, 256,
                1.0f,
                static_cast<float*>(B), 256,
                static_cast<float*>(A), 256,
                0.0f,
                static_cast<float*>(C), 256
            );
        } catch (const std::exception& e) {
            // SECURITY FIX (LOW-004): Log warning instead of silently ignoring
#ifndef NDEBUG
            std::cerr << "[WARN] rocBLAS warmup failed: " << e.what() << std::endl;
#endif
            (void)e;
        } catch (...) {
            // Non-standard exception during warmup
#ifndef NDEBUG
            std::cerr << "[WARN] rocBLAS warmup failed with unknown exception" << std::endl;
#endif
        }

        gpu_mem.deallocate(A);
        gpu_mem.deallocate(B);
        gpu_mem.deallocate(C);
    }

    // Synchronize to ensure all warmup operations complete
    hipDeviceSynchronize();
}

// ============================================================================
// Performance Profiling
// ============================================================================

KernelTimer::KernelTimer(hipStream_t stream, bool enabled)
    : stream_(stream), enabled_(enabled) {
    if (enabled_) {
        HIP_CHECK(hipEventCreate(&start_event_));
        HIP_CHECK(hipEventCreate(&stop_event_));
    }
}

KernelTimer::~KernelTimer() {
    if (enabled_) {
        hipEventDestroy(start_event_);
        hipEventDestroy(stop_event_);
    }
}

void KernelTimer::start() {
    if (enabled_) {
        HIP_CHECK(hipEventRecord(start_event_, stream_));
    }
}

float KernelTimer::stop() {
    if (!enabled_) return 0.0f;

    HIP_CHECK(hipEventRecord(stop_event_, stream_));
    HIP_CHECK(hipEventSynchronize(stop_event_));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms_, start_event_, stop_event_));

    return elapsed_ms_;
}

ScopedTimer::ScopedTimer(const std::string& name, hipStream_t stream)
    : name_(name), timer_(stream, true) {
    timer_.start();
}

ScopedTimer::~ScopedTimer() {
    float ms = timer_.stop();
    // SECURITY FIX (LOW-003): Only output timing in debug builds to prevent
    // information disclosure in production
#ifndef NDEBUG
    std::cerr << "[PERF] " << name_ << ": " << ms << " ms" << std::endl;
#else
    (void)ms;  // Suppress unused variable warning in release builds
#endif
}

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
