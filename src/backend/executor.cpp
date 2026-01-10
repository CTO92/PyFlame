// Executor - Runs computation graphs on CPU or CSL backend

#include "pyflame/ir/graph.hpp"
#include "pyflame/core/allocator.hpp"
#include "pyflame/backend/csl_codegen.hpp"

#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace pyflame::backend {

/// Execution backend type
enum class Backend {
    CPU,        // Reference CPU implementation
    SIMULATOR,  // Cerebras simulator
    HARDWARE,   // Actual Cerebras hardware
};

/// Result of graph execution
struct ExecutionResult {
    bool success = false;
    std::string error_message;

    // Timing (in milliseconds)
    double compile_time_ms = 0.0;
    double transfer_time_ms = 0.0;
    double compute_time_ms = 0.0;
    double total_time_ms = 0.0;

    // Output data (mapped by node ID)
    std::unordered_map<ir::NodeId, std::shared_ptr<uint8_t>> outputs;
};

/// Executor configuration
struct ExecutorConfig {
    Backend backend = Backend::CPU;
    bool enable_profiling = false;
    bool cache_compiled_kernels = true;
    std::string device_address = "localhost:9000";
};

/// Graph executor
class Executor {
public:
    explicit Executor(ExecutorConfig config = {}) : config_(config) {}

    /// Execute a computation graph
    ExecutionResult execute(
        ir::Graph& graph,
        const std::vector<ir::NodeId>& output_ids
    ) {
        switch (config_.backend) {
            case Backend::CPU:
                return execute_cpu(graph, output_ids);
            case Backend::SIMULATOR:
            case Backend::HARDWARE:
                return execute_csl(graph, output_ids);
        }
        return ExecutionResult{false, "Unknown backend"};
    }

private:
    ExecutorConfig config_;

    // CPU reference execution
    ExecutionResult execute_cpu(
        ir::Graph& graph,
        const std::vector<ir::NodeId>& output_ids
    ) {
        ExecutionResult result;
        result.success = true;

        try {
            // Get topological order
            auto topo = graph.topological_order();

            // Storage for intermediate results
            std::unordered_map<ir::NodeId, std::shared_ptr<uint8_t>> node_data;

            for (const auto& node : topo) {
                size_t bytes = node->output_spec().size_bytes();

                // Allocate output buffer
                auto data = std::shared_ptr<uint8_t>(
                    static_cast<uint8_t*>(Allocator::allocate(bytes)),
                    [](uint8_t* p) { Allocator::deallocate(p); }
                );

                // Handle different node types
                if (node->is_constant()) {
                    // Copy constant data
                    if (node->has_constant_data()) {
                        std::memcpy(data.get(), node->constant_data().data(), bytes);
                    } else {
                        std::memset(data.get(), 0, bytes);
                    }
                }
                else if (node->is_input() || node->is_parameter()) {
                    // Inputs should already have data
                    if (node->has_constant_data()) {
                        std::memcpy(data.get(), node->constant_data().data(), bytes);
                    }
                }
                else if (node->is_operation()) {
                    // Execute operation
                    execute_op_cpu(node, node_data, data);
                }

                node_data[node->id()] = data;
            }

            // Collect outputs
            for (auto id : output_ids) {
                if (node_data.count(id)) {
                    result.outputs[id] = node_data[id];
                }
            }
        }
        catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }

        return result;
    }

    void execute_op_cpu(
        const std::shared_ptr<ir::Node>& node,
        const std::unordered_map<ir::NodeId, std::shared_ptr<uint8_t>>& node_data,
        std::shared_ptr<uint8_t> output
    ) {
        const auto& inputs = node->inputs();
        auto op = node->op_type();
        int64_t numel = node->numel();

        // Get input data
        std::vector<const float*> input_ptrs;
        for (const auto& input : inputs) {
            auto it = node_data.find(input->id());
            if (it != node_data.end()) {
                input_ptrs.push_back(reinterpret_cast<const float*>(it->second.get()));
            } else {
                throw std::runtime_error("Input not found: " + std::to_string(input->id()));
            }
        }

        float* out = reinterpret_cast<float*>(output.get());

        // Execute based on operation type
        switch (op) {
            // Binary ops
            case ir::OpType::ADD:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = input_ptrs[0][i] + input_ptrs[1][i];
                }
                break;

            case ir::OpType::SUB:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = input_ptrs[0][i] - input_ptrs[1][i];
                }
                break;

            case ir::OpType::MUL:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = input_ptrs[0][i] * input_ptrs[1][i];
                }
                break;

            case ir::OpType::DIV:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = input_ptrs[0][i] / input_ptrs[1][i];
                }
                break;

            // Unary ops
            case ir::OpType::NEG:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = -input_ptrs[0][i];
                }
                break;

            case ir::OpType::ABS:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = std::abs(input_ptrs[0][i]);
                }
                break;

            case ir::OpType::SQRT:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = std::sqrt(input_ptrs[0][i]);
                }
                break;

            case ir::OpType::EXP:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = std::exp(input_ptrs[0][i]);
                }
                break;

            case ir::OpType::LOG:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = std::log(input_ptrs[0][i]);
                }
                break;

            case ir::OpType::SIN:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = std::sin(input_ptrs[0][i]);
                }
                break;

            case ir::OpType::COS:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = std::cos(input_ptrs[0][i]);
                }
                break;

            case ir::OpType::TANH:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = std::tanh(input_ptrs[0][i]);
                }
                break;

            // Activations
            case ir::OpType::RELU:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = std::max(0.0f, input_ptrs[0][i]);
                }
                break;

            case ir::OpType::SIGMOID:
                for (int64_t i = 0; i < numel; ++i) {
                    out[i] = 1.0f / (1.0f + std::exp(-input_ptrs[0][i]));
                }
                break;

            case ir::OpType::GELU:
                for (int64_t i = 0; i < numel; ++i) {
                    float x = input_ptrs[0][i];
                    out[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                }
                break;

            case ir::OpType::SILU:
                for (int64_t i = 0; i < numel; ++i) {
                    float x = input_ptrs[0][i];
                    out[i] = x / (1.0f + std::exp(-x));
                }
                break;

            // Reductions (simplified - full reduction only)
            case ir::OpType::SUM: {
                int64_t input_numel = inputs[0]->numel();
                float sum = 0.0f;
                for (int64_t i = 0; i < input_numel; ++i) {
                    sum += input_ptrs[0][i];
                }
                out[0] = sum;
                break;
            }

            case ir::OpType::MEAN: {
                int64_t input_numel = inputs[0]->numel();
                float sum = 0.0f;
                for (int64_t i = 0; i < input_numel; ++i) {
                    sum += input_ptrs[0][i];
                }
                out[0] = sum / static_cast<float>(input_numel);
                break;
            }

            case ir::OpType::MAX: {
                int64_t input_numel = inputs[0]->numel();
                float max_val = input_ptrs[0][0];
                for (int64_t i = 1; i < input_numel; ++i) {
                    max_val = std::max(max_val, input_ptrs[0][i]);
                }
                out[0] = max_val;
                break;
            }

            case ir::OpType::MIN: {
                int64_t input_numel = inputs[0]->numel();
                float min_val = input_ptrs[0][0];
                for (int64_t i = 1; i < input_numel; ++i) {
                    min_val = std::min(min_val, input_ptrs[0][i]);
                }
                out[0] = min_val;
                break;
            }

            // Matmul
            case ir::OpType::MATMUL: {
                auto a_shape = inputs[0]->shape();
                auto b_shape = inputs[1]->shape();

                if (a_shape.size() == 2 && b_shape.size() == 2) {
                    int64_t M = a_shape[0];
                    int64_t K = a_shape[1];
                    int64_t N = b_shape[1];

                    for (int64_t i = 0; i < M; ++i) {
                        for (int64_t j = 0; j < N; ++j) {
                            float sum = 0.0f;
                            for (int64_t k = 0; k < K; ++k) {
                                sum += input_ptrs[0][i * K + k] * input_ptrs[1][k * N + j];
                            }
                            out[i * N + j] = sum;
                        }
                    }
                } else {
                    throw std::runtime_error("Unsupported matmul dimensions");
                }
                break;
            }

            // Reshape (no-op for contiguous data)
            case ir::OpType::RESHAPE:
            case ir::OpType::VIEW: {
                int64_t input_numel = inputs[0]->numel();
                std::memcpy(out, input_ptrs[0], input_numel * sizeof(float));
                break;
            }

            // Transpose (2D only for now)
            case ir::OpType::TRANSPOSE: {
                auto shape = inputs[0]->shape();
                if (shape.size() == 2) {
                    int64_t M = shape[0];
                    int64_t N = shape[1];
                    for (int64_t i = 0; i < M; ++i) {
                        for (int64_t j = 0; j < N; ++j) {
                            out[j * M + i] = input_ptrs[0][i * N + j];
                        }
                    }
                } else {
                    throw std::runtime_error("Only 2D transpose supported");
                }
                break;
            }

            default:
                throw std::runtime_error("Unsupported operation: " + ir::op_type_name(op));
        }
    }

    // CSL backend execution (stub - requires Cerebras SDK)
    ExecutionResult execute_csl(
        ir::Graph& graph,
        const std::vector<ir::NodeId>& output_ids
    ) {
        ExecutionResult result;

        // Generate CSL code
        CSLCodeGenerator codegen;
        CodeGenOptions opts;
        opts.output_dir = "./pyflame_csl_output";

        auto gen_result = codegen.generate(graph, opts);
        if (!gen_result.success) {
            result.success = false;
            result.error_message = "CSL generation failed: " + gen_result.error_message;
            return result;
        }

        // In a real implementation, we would:
        // 1. Compile the CSL code using cslc
        // 2. Load the binary onto the simulator or hardware
        // 3. Transfer input data
        // 4. Execute
        // 5. Transfer output data back

        result.success = false;
        result.error_message = "CSL execution requires Cerebras SDK (not available)";
        return result;
    }
};

}  // namespace pyflame::backend
