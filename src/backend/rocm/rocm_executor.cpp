// ROCm Executor Implementation
// Phase 6: Full graph execution on AMD GPUs

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_executor.hpp"
#include "pyflame/backend/rocm/rocm_kernels.hpp"

#include <chrono>
#include <stdexcept>

namespace pyflame::backend {

// ExecutionResult definition (must match executor.cpp)
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

namespace rocm {

ROCmExecutor::ROCmExecutor(Config config) : config_(config) {
    // Set device
    HIP_CHECK(hipSetDevice(config_.device_id));

    // Create stream
    HIP_CHECK(hipStreamCreate(&stream_));

    // Initialize components
    ROCmMemoryManager::Config mem_config;
    mem_config.enable_pooling = true;
    memory_ = std::make_unique<ROCmMemoryManager>(mem_config);

    blas_ = std::make_unique<ROCmBLAS>();
    dnn_ = std::make_unique<ROCmDNN>();

    // Set stream for libraries
    blas_->set_stream(stream_);
    dnn_->set_stream(stream_);

    // Allocate workspace
    workspace_size_ = config_.workspace_size_mb * 1024 * 1024;
    workspace_ = memory_->allocate(workspace_size_);
}

ROCmExecutor::~ROCmExecutor() {
    if (workspace_) {
        memory_->deallocate(workspace_);
    }
    if (stream_) {
        hipStreamDestroy(stream_);
    }
}

ExecutionResult ROCmExecutor::execute(
    ir::Graph& graph,
    const std::vector<ir::NodeId>& output_ids
) {
    ExecutionResult result;
    result.success = true;

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Topological sort
        auto topo = graph.topological_order();

        // GPU tensor storage
        std::unordered_map<ir::NodeId, void*> gpu_data;
        std::unordered_map<ir::NodeId, size_t> gpu_sizes;

        auto transfer_start = std::chrono::high_resolution_clock::now();

        // Allocate and initialize GPU memory
        for (const auto& node : topo) {
            size_t bytes = node->output_spec().size_bytes();

            // Allocate GPU memory
            void* gpu_ptr = memory_->allocate(bytes);
            gpu_sizes[node->id()] = bytes;

            if (node->is_constant() || node->is_input() || node->is_parameter()) {
                // Copy data to GPU
                if (node->has_constant_data()) {
                    const auto& data = node->constant_data();
                    memory_->copy_host_to_device_async(
                        gpu_ptr, data.data(), data.size(), stream_);
                } else {
                    // Zero initialize
                    memory_->memset_async(gpu_ptr, 0, bytes, stream_);
                }
            }

            gpu_data[node->id()] = gpu_ptr;
        }

        auto transfer_end = std::chrono::high_resolution_clock::now();
        result.transfer_time_ms = std::chrono::duration<double, std::milli>(
            transfer_end - transfer_start).count();

        auto compute_start = std::chrono::high_resolution_clock::now();

        // Execute operations
        for (const auto& node : topo) {
            if (node->is_operation()) {
                execute_op(node, gpu_data, gpu_data[node->id()]);

                if (config_.sync_after_launch) {
                    HIP_CHECK(hipStreamSynchronize(stream_));
                }
            }
        }

        // Synchronize before reading results
        HIP_CHECK(hipStreamSynchronize(stream_));

        auto compute_end = std::chrono::high_resolution_clock::now();
        result.compute_time_ms = std::chrono::duration<double, std::milli>(
            compute_end - compute_start).count();

        // Copy outputs back to host
        for (auto id : output_ids) {
            auto it = gpu_data.find(id);
            if (it != gpu_data.end()) {
                size_t bytes = gpu_sizes[id];
                auto host_data = std::shared_ptr<uint8_t>(
                    new uint8_t[bytes],
                    std::default_delete<uint8_t[]>()
                );
                memory_->copy_device_to_host(host_data.get(), it->second, bytes);
                result.outputs[id] = host_data;
            }
        }

        // Free GPU memory
        for (auto& [id, ptr] : gpu_data) {
            memory_->deallocate(ptr);
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();

    return result;
}

void ROCmExecutor::execute_op(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    auto op = node->op_type();

    // Dispatch based on operation category
    if (ir::is_binary(op)) {
        // Check if it's a comparison operation
        if (op >= ir::OpType::EQ && op <= ir::OpType::LESS) {
            execute_comparison(node, gpu_data, output);
        } else {
            execute_elementwise_binary(node, gpu_data, output);
        }
        return;
    }

    if (ir::is_unary(op)) {
        execute_elementwise_unary(node, gpu_data, output);
        return;
    }

    if (ir::is_activation(op)) {
        execute_activation(node, gpu_data, output);
        return;
    }

    if (ir::is_reduction(op)) {
        execute_reduction(node, gpu_data, output);
        return;
    }

    // Specific operations
    switch (op) {
        case ir::OpType::MATMUL:
        case ir::OpType::BATCH_MATMUL:
            execute_matmul(node, gpu_data, output);
            break;

        case ir::OpType::CONV1D:
        case ir::OpType::CONV2D:
        case ir::OpType::CONV3D:
        case ir::OpType::CONV_TRANSPOSE2D:
            execute_conv2d(node, gpu_data, output);
            break;

        case ir::OpType::MAX_POOL1D:
        case ir::OpType::MAX_POOL2D:
        case ir::OpType::MAX_POOL3D:
        case ir::OpType::AVG_POOL1D:
        case ir::OpType::AVG_POOL2D:
        case ir::OpType::AVG_POOL3D:
        case ir::OpType::ADAPTIVE_AVG_POOL2D:
            execute_pooling(node, gpu_data, output);
            break;

        case ir::OpType::BATCH_NORM:
        case ir::OpType::LAYER_NORM:
        case ir::OpType::GROUP_NORM:
        case ir::OpType::INSTANCE_NORM:
            execute_batchnorm(node, gpu_data, output);
            break;

        case ir::OpType::SOFTMAX:
        case ir::OpType::LOG_SOFTMAX:
            execute_softmax(node, gpu_data, output);
            break;

        case ir::OpType::RESHAPE:
        case ir::OpType::VIEW:
        case ir::OpType::SQUEEZE:
        case ir::OpType::UNSQUEEZE:
        case ir::OpType::TRANSPOSE:
        case ir::OpType::SLICE:
        case ir::OpType::CONCAT:
        case ir::OpType::STACK:
        case ir::OpType::BROADCAST:
        case ir::OpType::COPY:
        case ir::OpType::CONTIGUOUS:
        case ir::OpType::LAYOUT_TRANSFORM:
            execute_shape_op(node, gpu_data, output);
            break;

        case ir::OpType::WHERE:
        case ir::OpType::CLAMP:
        case ir::OpType::MAXIMUM:
        case ir::OpType::MINIMUM:
            execute_conditional(node, gpu_data, output);
            break;

        case ir::OpType::NLL_LOSS:
        case ir::OpType::CROSS_ENTROPY_LOSS:
        case ir::OpType::MSE_LOSS:
        case ir::OpType::L1_LOSS:
        case ir::OpType::BCE_LOSS:
        case ir::OpType::BCE_WITH_LOGITS_LOSS:
        case ir::OpType::KL_DIV_LOSS:
        case ir::OpType::SMOOTH_L1_LOSS:
            execute_loss(node, gpu_data, output);
            break;

        case ir::OpType::EMBEDDING:
        case ir::OpType::GATHER:
            throw std::runtime_error("Embedding/Gather not yet implemented for ROCm");

        case ir::OpType::DROPOUT:
            throw std::runtime_error("Dropout not yet implemented for ROCm");

        default:
            throw std::runtime_error(
                "Unsupported operation for ROCm: " + ir::op_type_name(op));
    }
}

void ROCmExecutor::execute_matmul(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.size() < 2) {
        throw std::runtime_error("MATMUL requires at least 2 inputs, got " +
                                 std::to_string(inputs.size()));
    }

    auto A = gpu_data.at(inputs[0]->id());
    auto B = gpu_data.at(inputs[1]->id());

    auto a_shape = inputs[0]->shape();
    auto b_shape = inputs[1]->shape();
    auto dtype = node->output_spec().dtype();

    if (node->op_type() == ir::OpType::MATMUL && a_shape.size() == 2) {
        // Simple 2D GEMM: C = A @ B
        int64_t M = a_shape[0];
        int64_t K = a_shape[1];
        int64_t N = b_shape[1];

        blas_->gemm(
            dtype,
            TransposeOp::None, TransposeOp::None,
            M, N, K,
            1.0f,           // alpha
            A, K,           // A, lda
            B, N,           // B, ldb
            0.0f,           // beta
            output, N       // C, ldc
        );
    } else if (node->op_type() == ir::OpType::BATCH_MATMUL || a_shape.size() == 3) {
        // Batched GEMM
        int64_t batch = a_shape[0];
        int64_t M = a_shape[1];
        int64_t K = a_shape[2];
        int64_t N = b_shape[2];

        blas_->gemm_strided_batched(
            dtype,
            TransposeOp::None, TransposeOp::None,
            M, N, K,
            1.0f,
            A, K, M * K,    // A, lda, stride_a
            B, N, K * N,    // B, ldb, stride_b
            0.0f,
            output, N, M * N, // C, ldc, stride_c
            batch
        );
    } else {
        throw std::runtime_error("Unsupported matmul configuration");
    }
}

void ROCmExecutor::execute_conv2d(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.size() < 2) {
        throw std::runtime_error("CONV2D requires at least 2 inputs (input, weight), got " +
                                 std::to_string(inputs.size()));
    }

    auto input = gpu_data.at(inputs[0]->id());
    auto weight = gpu_data.at(inputs[1]->id());

    auto input_shape = inputs[0]->shape();
    auto weight_shape = inputs[1]->shape();
    auto output_shape = node->shape();
    auto dtype = node->output_spec().dtype();

    // Get convolution parameters from node attributes
    int pad_h = node->get_attr<int>("padding_h", 0);
    int pad_w = node->get_attr<int>("padding_w", 0);
    int stride_h = node->get_attr<int>("stride_h", 1);
    int stride_w = node->get_attr<int>("stride_w", 1);
    int dilation_h = node->get_attr<int>("dilation_h", 1);
    int dilation_w = node->get_attr<int>("dilation_w", 1);
    int groups = node->get_attr<int>("groups", 1);

    ConvParams params = ConvParams::Conv2D(
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, groups);

    // Get workspace size
    size_t ws_size = dnn_->conv2d_forward_workspace_size(
        dtype, input_shape, weight_shape, output_shape, params);

    // Use preallocated workspace if sufficient, otherwise allocate temporary
    void* ws = workspace_;
    bool temp_workspace = false;
    if (ws_size > workspace_size_) {
        ws = memory_->allocate(ws_size);
        temp_workspace = true;
    }

    dnn_->conv2d_forward(
        dtype, input_shape, input, weight_shape, weight,
        output_shape, output, params, ws, ws_size);

    if (temp_workspace) {
        memory_->deallocate(ws);
    }
}

void ROCmExecutor::execute_pooling(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.empty()) {
        throw std::runtime_error("POOLING requires at least 1 input");
    }

    auto input = gpu_data.at(inputs[0]->id());

    auto input_shape = inputs[0]->shape();
    auto output_shape = node->shape();
    auto dtype = node->output_spec().dtype();
    auto op = node->op_type();

    // Get pooling parameters
    int kernel_h = node->get_attr<int>("kernel_h", 2);
    int kernel_w = node->get_attr<int>("kernel_w", 2);
    int pad_h = node->get_attr<int>("padding_h", 0);
    int pad_w = node->get_attr<int>("padding_w", 0);
    int stride_h = node->get_attr<int>("stride_h", kernel_h);
    int stride_w = node->get_attr<int>("stride_w", kernel_w);
    bool ceil_mode = node->get_attr<bool>("ceil_mode", false);

    PoolParams params = PoolParams::Pool2D(
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, ceil_mode);

    dnn_->pooling2d_forward(op, dtype, input_shape, input, output_shape, output, params);
}

void ROCmExecutor::execute_batchnorm(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: BatchNorm requires exactly 5 inputs: input, weight (gamma), bias (beta), running_mean, running_var
    if (inputs.size() < 5) {
        throw std::runtime_error("BatchNorm requires 5 inputs (input, gamma, beta, running_mean, running_var), got " +
                                 std::to_string(inputs.size()));
    }

    auto input = gpu_data.at(inputs[0]->id());
    auto input_shape = inputs[0]->shape();
    auto dtype = node->output_spec().dtype();

    auto scale = gpu_data.at(inputs[1]->id());      // gamma
    auto bias = gpu_data.at(inputs[2]->id());       // beta
    auto running_mean = gpu_data.at(inputs[3]->id());
    auto running_var = gpu_data.at(inputs[4]->id());

    double epsilon = node->get_attr<double>("epsilon", 1e-5);
    double momentum = node->get_attr<double>("momentum", 0.1);
    bool training = node->get_attr<bool>("training", false);

    BatchNormParams params;
    params.epsilon = epsilon;
    params.momentum = momentum;
    params.training = training;

    if (training) {
        // Allocate temporary buffers for saved mean/var
        size_t channel_bytes = static_cast<size_t>(input_shape[1]) * sizeof(float);
        void* saved_mean = memory_->allocate(channel_bytes);
        void* saved_inv_var = memory_->allocate(channel_bytes);

        dnn_->batchnorm_forward_training(
            dtype, input_shape, input, output, scale, bias,
            const_cast<void*>(running_mean), const_cast<void*>(running_var),
            saved_mean, saved_inv_var, params);

        memory_->deallocate(saved_mean);
        memory_->deallocate(saved_inv_var);
    } else {
        dnn_->batchnorm_forward_inference(
            dtype, input_shape, input, output, scale, bias,
            running_mean, running_var, params);
    }
}

void ROCmExecutor::execute_activation(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.empty()) {
        throw std::runtime_error("Activation requires at least 1 input");
    }

    auto input = gpu_data.at(inputs[0]->id());
    auto op = node->op_type();
    auto dtype = node->output_spec().dtype();
    auto shape = node->shape();
    int64_t numel = node->numel();

    // GELU and SiLU use custom kernels
    if (op == ir::OpType::GELU) {
        kernels::launch_gelu_forward(
            static_cast<float*>(output),
            static_cast<const float*>(input),
            numel, stream_);
    } else if (op == ir::OpType::SILU) {
        kernels::launch_silu_forward(
            static_cast<float*>(output),
            static_cast<const float*>(input),
            numel, stream_);
    } else {
        // Use MIOpen for ReLU, Sigmoid, Tanh
        dnn_->activation_forward(op, dtype, shape, input, output);
    }
}

void ROCmExecutor::execute_softmax(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.empty()) {
        throw std::runtime_error("Softmax requires at least 1 input");
    }

    auto input = gpu_data.at(inputs[0]->id());
    auto dtype = node->output_spec().dtype();
    auto shape = node->shape();
    auto op = node->op_type();

    int axis = node->get_attr<int>("axis", -1);
    bool log_softmax = (op == ir::OpType::LOG_SOFTMAX);

    dnn_->softmax_forward(dtype, shape, input, output, axis, log_softmax);
}

void ROCmExecutor::execute_reduction(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.empty()) {
        throw std::runtime_error("Reduction requires at least 1 input");
    }

    auto input = gpu_data.at(inputs[0]->id());
    auto op = node->op_type();
    auto dtype = node->output_spec().dtype();
    auto input_shape = inputs[0]->shape();
    auto output_shape = node->shape();

    // Get reduction dimensions
    std::vector<int> reduce_dims;
    if (node->has_attr("dim")) {
        reduce_dims = node->get_attr<std::vector<int>>("dim");
    } else {
        // Default: reduce all dimensions
        for (int i = 0; i < static_cast<int>(input_shape.size()); ++i) {
            reduce_dims.push_back(i);
        }
    }

    // Special case for PROD which uses custom kernel
    if (op == ir::OpType::PROD) {
        kernels::launch_reduce_prod(
            static_cast<float*>(output),
            static_cast<const float*>(input),
            inputs[0]->numel(), stream_);
        return;
    }

    // Use MIOpen for SUM, MEAN, MAX, MIN
    size_t ws_size = dnn_->reduce_workspace_size(op, dtype, input_shape, output_shape);

    void* ws = workspace_;
    bool temp_workspace = false;
    if (ws_size > workspace_size_) {
        ws = memory_->allocate(ws_size);
        temp_workspace = true;
    }

    dnn_->reduce(op, dtype, input_shape, input, output_shape, output,
                 reduce_dims, ws, ws_size);

    if (temp_workspace) {
        memory_->deallocate(ws);
    }
}

void ROCmExecutor::execute_elementwise_binary(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.size() < 2) {
        throw std::runtime_error("Binary operation requires 2 inputs, got " +
                                 std::to_string(inputs.size()));
    }

    auto a = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
    auto b = static_cast<const float*>(gpu_data.at(inputs[1]->id()));
    auto out = static_cast<float*>(output);
    int64_t numel = node->numel();

    kernels::launch_elementwise_binary(
        node->op_type(), out, a, b, numel, stream_);
}

void ROCmExecutor::execute_elementwise_unary(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.empty()) {
        throw std::runtime_error("Unary operation requires at least 1 input");
    }

    auto input = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
    auto out = static_cast<float*>(output);
    int64_t numel = node->numel();

    kernels::launch_elementwise_unary(
        node->op_type(), out, input, numel, stream_);
}

void ROCmExecutor::execute_comparison(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.size() < 2) {
        throw std::runtime_error("Comparison operation requires 2 inputs, got " +
                                 std::to_string(inputs.size()));
    }

    auto a = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
    auto b = static_cast<const float*>(gpu_data.at(inputs[1]->id()));
    auto out = static_cast<float*>(output);
    int64_t numel = node->numel();

    kernels::launch_comparison(
        node->op_type(), out, a, b, numel, stream_);
}

void ROCmExecutor::execute_conditional(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();
    auto op = node->op_type();
    int64_t numel = node->numel();

    if (op == ir::OpType::WHERE) {
        // SECURITY: WHERE requires exactly 3 inputs: condition, x, y
        if (inputs.size() < 3) {
            throw std::runtime_error("WHERE requires 3 inputs (condition, x, y), got " +
                                     std::to_string(inputs.size()));
        }

        auto condition = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
        auto x = static_cast<const float*>(gpu_data.at(inputs[1]->id()));
        auto y = static_cast<const float*>(gpu_data.at(inputs[2]->id()));
        auto out = static_cast<float*>(output);

        kernels::launch_where(out, condition, x, y, numel, stream_);
    } else if (op == ir::OpType::CLAMP) {
        // SECURITY: CLAMP requires at least 1 input
        if (inputs.empty()) {
            throw std::runtime_error("CLAMP requires at least 1 input");
        }

        auto input = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
        auto out = static_cast<float*>(output);

        float min_val = node->get_attr<float>("min", -std::numeric_limits<float>::infinity());
        float max_val = node->get_attr<float>("max", std::numeric_limits<float>::infinity());

        kernels::launch_clamp(out, input, min_val, max_val, numel, stream_);
    } else if (op == ir::OpType::MAXIMUM || op == ir::OpType::MINIMUM) {
        // SECURITY: MAXIMUM/MINIMUM requires 2 inputs
        if (inputs.size() < 2) {
            throw std::runtime_error(ir::op_type_name(op) + " requires 2 inputs, got " +
                                     std::to_string(inputs.size()));
        }

        // Element-wise maximum/minimum
        auto a = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
        auto b = static_cast<const float*>(gpu_data.at(inputs[1]->id()));
        auto out = static_cast<float*>(output);

        kernels::launch_elementwise_binary(op, out, a, b, numel, stream_);
    } else {
        throw std::runtime_error("Unknown conditional op: " + ir::op_type_name(op));
    }
}

void ROCmExecutor::execute_shape_op(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();

    // SECURITY: Validate input count before access
    if (inputs.empty()) {
        throw std::runtime_error("Shape operation requires at least 1 input");
    }

    auto input = gpu_data.at(inputs[0]->id());
    auto op = node->op_type();
    size_t bytes = node->output_spec().size_bytes();

    switch (op) {
        case ir::OpType::RESHAPE:
        case ir::OpType::VIEW:
        case ir::OpType::SQUEEZE:
        case ir::OpType::UNSQUEEZE:
        case ir::OpType::CONTIGUOUS:
        case ir::OpType::COPY:
        case ir::OpType::LAYOUT_TRANSFORM:
            // These are just memory copies (same data, different view)
            memory_->copy_device_to_device_async(output, input, bytes, stream_);
            break;

        case ir::OpType::TRANSPOSE: {
            // 2D transpose using rocBLAS geam
            auto shape = inputs[0]->shape();
            if (shape.size() == 2) {
                int64_t M = shape[0];
                int64_t N = shape[1];
                blas_->geam(
                    node->output_spec().dtype(),
                    TransposeOp::Transpose,
                    TransposeOp::None,
                    N, M,  // output is N x M
                    1.0f,
                    input, N,  // input is M x N, lda = N
                    0.0f,
                    nullptr, 0,
                    output, M  // output is N x M, ldc = M
                );
            } else {
                throw std::runtime_error("Multi-dim transpose not yet implemented for ROCm");
            }
            break;
        }

        case ir::OpType::CONCAT:
        case ir::OpType::STACK:
        case ir::OpType::SLICE:
        case ir::OpType::BROADCAST:
            throw std::runtime_error(
                ir::op_type_name(op) + " not yet implemented for ROCm");

        default:
            throw std::runtime_error(
                "Unsupported shape op: " + ir::op_type_name(op));
    }
}

void ROCmExecutor::execute_loss(
    const std::shared_ptr<ir::Node>& node,
    const std::unordered_map<ir::NodeId, void*>& gpu_data,
    void* output
) {
    const auto& inputs = node->inputs();
    auto op = node->op_type();

    // SECURITY: All loss functions require at least 2 inputs (predictions, targets)
    if (inputs.size() < 2) {
        throw std::runtime_error("Loss function requires at least 2 inputs (predictions, targets), got " +
                                 std::to_string(inputs.size()));
    }

    switch (op) {
        case ir::OpType::MSE_LOSS: {
            auto predictions = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
            auto targets = static_cast<const float*>(gpu_data.at(inputs[1]->id()));
            auto out = static_cast<float*>(output);
            int64_t numel = inputs[0]->numel();

            kernels::launch_mse_loss(out, predictions, targets, numel, stream_);
            break;
        }

        case ir::OpType::BCE_LOSS: {
            auto predictions = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
            auto targets = static_cast<const float*>(gpu_data.at(inputs[1]->id()));
            auto out = static_cast<float*>(output);
            int64_t numel = inputs[0]->numel();

            kernels::launch_bce_loss(out, predictions, targets, numel, stream_);
            break;
        }

        case ir::OpType::CROSS_ENTROPY_LOSS: {
            auto logits = static_cast<const float*>(gpu_data.at(inputs[0]->id()));
            auto targets = static_cast<const int32_t*>(gpu_data.at(inputs[1]->id()));
            auto out = static_cast<float*>(output);

            auto logits_shape = inputs[0]->shape();
            int64_t N = logits_shape[0];  // batch size
            int64_t C = logits_shape[1];  // num classes

            kernels::launch_cross_entropy_loss(out, logits, targets, N, C, stream_);
            break;
        }

        case ir::OpType::NLL_LOSS:
        case ir::OpType::L1_LOSS:
        case ir::OpType::BCE_WITH_LOGITS_LOSS:
        case ir::OpType::KL_DIV_LOSS:
        case ir::OpType::SMOOTH_L1_LOSS:
            throw std::runtime_error(
                ir::op_type_name(op) + " not yet implemented for ROCm");

        default:
            throw std::runtime_error(
                "Unknown loss function: " + ir::op_type_name(op));
    }
}

DeviceInfo ROCmExecutor::get_device_info() const {
    return pyflame::backend::rocm::get_device_info(config_.device_id);
}

MemoryStats ROCmExecutor::get_memory_stats() const {
    return memory_->get_stats();
}

}  // namespace rocm
}  // namespace pyflame::backend

#endif  // PYFLAME_HAS_ROCM
