#pragma once

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_backend.hpp"
#include "pyflame/backend/rocm/rocm_memory.hpp"
#include "pyflame/backend/rocm/rocm_blas.hpp"
#include "pyflame/backend/rocm/rocm_dnn.hpp"
#include "pyflame/ir/graph.hpp"

#include <unordered_map>
#include <memory>

namespace pyflame::backend {

// Forward declare ExecutionResult (defined in executor.cpp)
struct ExecutionResult;

namespace rocm {

/// ROCm graph executor
///
/// Executes computation graphs on AMD GPUs using ROCm libraries:
/// - rocBLAS for matrix operations
/// - MIOpen for DNN operations (convolution, pooling, batch norm)
/// - Custom HIP kernels for activations and elementwise ops
///
/// Example:
/// @code
/// ROCmExecutor::Config config;
/// config.device_id = 0;
/// config.workspace_size_mb = 256;
///
/// ROCmExecutor executor(config);
/// auto result = executor.execute(graph, output_ids);
/// @endcode
class ROCmExecutor {
public:
    /// Configuration options
    struct Config {
        int device_id = 0;               ///< GPU device to use
        bool enable_profiling = false;   ///< Enable timing measurements
        bool sync_after_launch = false;  ///< Sync after each kernel (debug)
        size_t workspace_size_mb = 256;  ///< Workspace size for algorithms

        Config() = default;
    };

    /// Construct executor with default configuration
    ROCmExecutor();

    /// Construct executor with configuration
    /// @param config Executor configuration options
    explicit ROCmExecutor(const Config& config);

    /// Destructor - releases GPU resources
    ~ROCmExecutor();

    // Disable copy
    ROCmExecutor(const ROCmExecutor&) = delete;
    ROCmExecutor& operator=(const ROCmExecutor&) = delete;

    /// Execute a computation graph
    ///
    /// Executes the graph on the GPU and returns results for specified outputs.
    /// Memory is automatically managed - allocated before execution and freed after.
    ///
    /// @param graph The computation graph to execute
    /// @param output_ids Node IDs whose outputs should be returned
    /// @return ExecutionResult containing success status, timing, and output data
    ExecutionResult execute(
        ir::Graph& graph,
        const std::vector<ir::NodeId>& output_ids
    );

    /// Get device info for the current device
    /// @return DeviceInfo struct with GPU details
    DeviceInfo get_device_info() const;

    /// Get memory statistics
    /// @return MemoryStats with allocation info
    MemoryStats get_memory_stats() const;

private:
    Config config_;
    hipStream_t stream_;
    std::unique_ptr<ROCmMemoryManager> memory_;
    std::unique_ptr<ROCmBLAS> blas_;
    std::unique_ptr<ROCmDNN> dnn_;

    // Workspace for convolution algorithms, etc.
    void* workspace_;
    size_t workspace_size_;

    /// Execute a single operation
    void execute_op(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    // Operation-specific dispatchers

    /// Execute matrix multiplication (GEMM, batched GEMM)
    void execute_matmul(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute 2D convolution
    void execute_conv2d(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute pooling operations (max, avg)
    void execute_pooling(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute batch normalization
    void execute_batchnorm(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute activation functions (ReLU, GELU, SiLU, etc.)
    void execute_activation(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute softmax/log_softmax
    void execute_softmax(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute reduction operations (sum, mean, max, min)
    void execute_reduction(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute elementwise binary operations
    void execute_elementwise_binary(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute elementwise unary operations
    void execute_elementwise_unary(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute comparison operations (eq, ne, lt, etc.)
    void execute_comparison(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute shape operations (reshape, transpose, etc.)
    void execute_shape_op(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute loss function operations
    void execute_loss(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );

    /// Execute conditional operations (where, clamp)
    void execute_conditional(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, void*>& gpu_data,
        void* output
    );
};

}  // namespace rocm
}  // namespace pyflame::backend

#endif  // PYFLAME_HAS_ROCM
