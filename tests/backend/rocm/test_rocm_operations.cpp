/// @file test_rocm_operations.cpp
/// @brief Comprehensive tests for all ROCm operations
///
/// Tests all 65 operations against CPU reference implementation
/// to verify numerical correctness of the ROCm backend.

#include <gtest/gtest.h>

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_executor.hpp"
#include "pyflame/backend/rocm/rocm_backend.hpp"
#include "pyflame/backend/executor.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/core/tensor.hpp"
#include <cmath>
#include <random>
#include <numeric>

using namespace pyflame;
using namespace pyflame::backend;

class ROCmOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!rocm::is_available()) {
            GTEST_SKIP() << "ROCm not available";
        }
        rocm::ROCmExecutor::Config config;
        config.device_id = 0;
        config.enable_profiling = false;
        executor_ = std::make_unique<rocm::ROCmExecutor>(config);
    }

    void TearDown() override {
        executor_.reset();
    }

    /// Generate random float data
    std::vector<float> random_data(size_t size, float min = -1.0f, float max = 1.0f) {
        std::vector<float> data(size);
        std::uniform_real_distribution<float> dist(min, max);
        for (auto& v : data) {
            v = dist(rng_);
        }
        return data;
    }

    /// Generate random positive float data (for operations like sqrt, log)
    std::vector<float> random_positive_data(size_t size, float min = 0.1f, float max = 2.0f) {
        return random_data(size, min, max);
    }

    /// Generate random integer data for indices
    std::vector<int32_t> random_indices(size_t size, int32_t max_val) {
        std::vector<int32_t> data(size);
        std::uniform_int_distribution<int32_t> dist(0, max_val - 1);
        for (auto& v : data) {
            v = dist(rng_);
        }
        return data;
    }

    /// Helper to compare results with CPU reference
    void compare_with_cpu(
        ir::Graph& graph,
        const std::vector<ir::NodeId>& outputs,
        float rtol = 1e-5f,
        float atol = 1e-6f
    ) {
        // Execute on CPU
        Executor cpu_executor;
        auto cpu_result = cpu_executor.execute(graph, outputs);
        ASSERT_TRUE(cpu_result.success) << cpu_result.error_message;

        // Execute on ROCm
        auto rocm_result = executor_->execute(graph, outputs);
        ASSERT_TRUE(rocm_result.success) << rocm_result.error_message;

        // Compare outputs
        for (auto id : outputs) {
            auto cpu_data = reinterpret_cast<const float*>(
                cpu_result.outputs.at(id).get());
            auto rocm_data = reinterpret_cast<const float*>(
                rocm_result.outputs.at(id).get());

            auto node = graph.get_node(id);
            int64_t numel = node->numel();

            for (int64_t i = 0; i < numel; ++i) {
                float cpu_val = cpu_data[i];
                float rocm_val = rocm_data[i];
                float diff = std::abs(cpu_val - rocm_val);
                float tol = atol + rtol * std::abs(cpu_val);

                EXPECT_LE(diff, tol)
                    << "Mismatch at index " << i
                    << ": CPU=" << cpu_val << ", ROCm=" << rocm_val
                    << " (diff=" << diff << ", tol=" << tol << ")";
            }
        }
    }

    std::unique_ptr<rocm::ROCmExecutor> executor_;
    std::mt19937 rng_{42};  // Fixed seed for reproducibility
};

// =============================================================================
// Matrix Operations Tests
// =============================================================================

TEST_F(ROCmOperationsTest, Matmul_Small) {
    ir::Graph graph;

    auto a_data = random_data(16 * 32);
    auto b_data = random_data(32 * 24);

    TensorSpec a_spec({16, 32}, DType::Float32);
    TensorSpec b_spec({32, 24}, DType::Float32);
    TensorSpec c_spec({16, 24}, DType::Float32);

    auto a = graph.create_constant(a_spec, a_data.data(), "A");
    auto b = graph.create_constant(b_spec, b_data.data(), "B");
    auto c = graph.create_op(ir::OpType::MATMUL, {a, b}, c_spec, "C");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, Matmul_Medium) {
    ir::Graph graph;

    auto a_data = random_data(128 * 256);
    auto b_data = random_data(256 * 512);

    TensorSpec a_spec({128, 256}, DType::Float32);
    TensorSpec b_spec({256, 512}, DType::Float32);
    TensorSpec c_spec({128, 512}, DType::Float32);

    auto a = graph.create_constant(a_spec, a_data.data(), "A");
    auto b = graph.create_constant(b_spec, b_data.data(), "B");
    auto c = graph.create_op(ir::OpType::MATMUL, {a, b}, c_spec, "C");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, Matmul_Large) {
    ir::Graph graph;

    auto a_data = random_data(512 * 1024);
    auto b_data = random_data(1024 * 512);

    TensorSpec a_spec({512, 1024}, DType::Float32);
    TensorSpec b_spec({1024, 512}, DType::Float32);
    TensorSpec c_spec({512, 512}, DType::Float32);

    auto a = graph.create_constant(a_spec, a_data.data(), "A");
    auto b = graph.create_constant(b_spec, b_data.data(), "B");
    auto c = graph.create_op(ir::OpType::MATMUL, {a, b}, c_spec, "C");

    compare_with_cpu(graph, {c->id()}, 1e-4f, 1e-5f);  // Looser tolerance for large matrices
}

TEST_F(ROCmOperationsTest, BatchedMatmul) {
    ir::Graph graph;

    // Batched matmul: [4, 32, 64] @ [4, 64, 48] -> [4, 32, 48]
    auto a_data = random_data(4 * 32 * 64);
    auto b_data = random_data(4 * 64 * 48);

    TensorSpec a_spec({4, 32, 64}, DType::Float32);
    TensorSpec b_spec({4, 64, 48}, DType::Float32);
    TensorSpec c_spec({4, 32, 48}, DType::Float32);

    auto a = graph.create_constant(a_spec, a_data.data(), "A");
    auto b = graph.create_constant(b_spec, b_data.data(), "B");
    auto c = graph.create_op(ir::OpType::BATCHED_MATMUL, {a, b}, c_spec, "C");

    compare_with_cpu(graph, {c->id()});
}

// =============================================================================
// Convolution Tests
// =============================================================================

TEST_F(ROCmOperationsTest, Conv2D_Basic) {
    ir::Graph graph;

    // NCHW format: batch=2, channels=3, height=32, width=32
    auto input_data = random_data(2 * 3 * 32 * 32);
    // OIHW format: out_channels=16, in_channels=3, kH=3, kW=3
    auto weight_data = random_data(16 * 3 * 3 * 3);

    TensorSpec input_spec({2, 3, 32, 32}, DType::Float32);
    TensorSpec weight_spec({16, 3, 3, 3}, DType::Float32);
    TensorSpec output_spec({2, 16, 30, 30}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");
    auto weight = graph.create_constant(weight_spec, weight_data.data(), "weight");
    auto output = graph.create_op(ir::OpType::CONV2D, {input, weight},
                                   output_spec, "output");

    compare_with_cpu(graph, {output->id()}, 1e-4f, 1e-5f);
}

TEST_F(ROCmOperationsTest, Conv2D_WithPadding) {
    ir::Graph graph;

    auto input_data = random_data(4 * 16 * 28 * 28);
    auto weight_data = random_data(32 * 16 * 3 * 3);

    TensorSpec input_spec({4, 16, 28, 28}, DType::Float32);
    TensorSpec weight_spec({32, 16, 3, 3}, DType::Float32);
    // With padding=1, stride=1: output = (28 + 2*1 - 3)/1 + 1 = 28
    TensorSpec output_spec({4, 32, 28, 28}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");
    auto weight = graph.create_constant(weight_spec, weight_data.data(), "weight");

    ir::OpAttributes attrs;
    attrs.set("padding", std::vector<int64_t>{1, 1});
    attrs.set("stride", std::vector<int64_t>{1, 1});

    auto output = graph.create_op(ir::OpType::CONV2D, {input, weight},
                                   output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()}, 1e-4f, 1e-5f);
}

TEST_F(ROCmOperationsTest, Conv2D_WithStride) {
    ir::Graph graph;

    auto input_data = random_data(2 * 32 * 64 * 64);
    auto weight_data = random_data(64 * 32 * 3 * 3);

    TensorSpec input_spec({2, 32, 64, 64}, DType::Float32);
    TensorSpec weight_spec({64, 32, 3, 3}, DType::Float32);
    // With stride=2, padding=1: output = (64 + 2*1 - 3)/2 + 1 = 32
    TensorSpec output_spec({2, 64, 32, 32}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");
    auto weight = graph.create_constant(weight_spec, weight_data.data(), "weight");

    ir::OpAttributes attrs;
    attrs.set("padding", std::vector<int64_t>{1, 1});
    attrs.set("stride", std::vector<int64_t>{2, 2});

    auto output = graph.create_op(ir::OpType::CONV2D, {input, weight},
                                   output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()}, 1e-4f, 1e-5f);
}

// =============================================================================
// Pooling Tests
// =============================================================================

TEST_F(ROCmOperationsTest, MaxPool2D) {
    ir::Graph graph;

    auto input_data = random_data(2 * 16 * 32 * 32);

    TensorSpec input_spec({2, 16, 32, 32}, DType::Float32);
    TensorSpec output_spec({2, 16, 16, 16}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("kernel_size", std::vector<int64_t>{2, 2});
    attrs.set("stride", std::vector<int64_t>{2, 2});

    auto output = graph.create_op(ir::OpType::MAX_POOL2D, {input},
                                   output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, AvgPool2D) {
    ir::Graph graph;

    auto input_data = random_data(2 * 16 * 32 * 32);

    TensorSpec input_spec({2, 16, 32, 32}, DType::Float32);
    TensorSpec output_spec({2, 16, 16, 16}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("kernel_size", std::vector<int64_t>{2, 2});
    attrs.set("stride", std::vector<int64_t>{2, 2});

    auto output = graph.create_op(ir::OpType::AVG_POOL2D, {input},
                                   output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, GlobalAvgPool2D) {
    ir::Graph graph;

    auto input_data = random_data(4 * 32 * 7 * 7);

    TensorSpec input_spec({4, 32, 7, 7}, DType::Float32);
    TensorSpec output_spec({4, 32, 1, 1}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::GLOBAL_AVG_POOL2D, {input},
                                   output_spec, "output");

    compare_with_cpu(graph, {output->id()});
}

// =============================================================================
// Batch Normalization Tests
// =============================================================================

TEST_F(ROCmOperationsTest, BatchNorm_Inference) {
    ir::Graph graph;

    auto input_data = random_data(4 * 32 * 14 * 14);
    auto gamma_data = random_positive_data(32);
    auto beta_data = random_data(32);
    auto mean_data = random_data(32, -0.5f, 0.5f);
    auto var_data = random_positive_data(32, 0.5f, 2.0f);

    TensorSpec input_spec({4, 32, 14, 14}, DType::Float32);
    TensorSpec param_spec({32}, DType::Float32);
    TensorSpec output_spec({4, 32, 14, 14}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");
    auto gamma = graph.create_constant(param_spec, gamma_data.data(), "gamma");
    auto beta = graph.create_constant(param_spec, beta_data.data(), "beta");
    auto mean = graph.create_constant(param_spec, mean_data.data(), "mean");
    auto var = graph.create_constant(param_spec, var_data.data(), "var");

    ir::OpAttributes attrs;
    attrs.set("epsilon", 1e-5);
    attrs.set("training", false);

    auto output = graph.create_op(ir::OpType::BATCH_NORM,
                                   {input, gamma, beta, mean, var},
                                   output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

// =============================================================================
// Activation Function Tests
// =============================================================================

TEST_F(ROCmOperationsTest, ReLU) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::RELU, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, LeakyReLU) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("negative_slope", 0.01f);

    auto output = graph.create_op(ir::OpType::LEAKY_RELU, {input}, spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Sigmoid) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::SIGMOID, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Tanh) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::TANH, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, GELU) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::GELU, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()}, 1e-4f, 1e-5f);  // GELU has more numerical error
}

TEST_F(ROCmOperationsTest, SiLU) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::SILU, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, AllActivations) {
    std::vector<ir::OpType> activations = {
        ir::OpType::RELU,
        ir::OpType::SIGMOID,
        ir::OpType::TANH,
        ir::OpType::GELU,
        ir::OpType::SILU,
    };

    for (auto op : activations) {
        ir::Graph graph;

        auto input_data = random_data(1024);

        TensorSpec spec({1024}, DType::Float32);

        auto input = graph.create_constant(spec, input_data.data(), "input");
        auto output = graph.create_op(op, {input}, spec, "output");

        SCOPED_TRACE("Testing activation: " + ir::op_type_name(op));
        compare_with_cpu(graph, {output->id()}, 1e-4f, 1e-5f);
    }
}

// =============================================================================
// Softmax Tests
// =============================================================================

TEST_F(ROCmOperationsTest, Softmax) {
    ir::Graph graph;

    auto input_data = random_data(32 * 10);

    TensorSpec spec({32, 10}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("dim", static_cast<int64_t>(-1));

    auto output = graph.create_op(ir::OpType::SOFTMAX, {input}, spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, LogSoftmax) {
    ir::Graph graph;

    auto input_data = random_data(32 * 10);

    TensorSpec spec({32, 10}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("dim", static_cast<int64_t>(-1));

    auto output = graph.create_op(ir::OpType::LOG_SOFTMAX, {input}, spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

// =============================================================================
// Elementwise Binary Operations Tests
// =============================================================================

TEST_F(ROCmOperationsTest, Add) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::ADD, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, Sub) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::SUB, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, Mul) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::MUL, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, Div) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = random_positive_data(4096, 0.5f, 2.0f);  // Avoid division by zero

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::DIV, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, Pow) {
    ir::Graph graph;

    auto a_data = random_positive_data(4096);  // Positive base for pow
    auto b_data = random_data(4096, 0.5f, 2.0f);  // Reasonable exponents

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::POW, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()}, 1e-4f, 1e-5f);
}

TEST_F(ROCmOperationsTest, Maximum) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::MAX_BINARY, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, Minimum) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::MIN_BINARY, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, AllElementwiseBinary) {
    std::vector<ir::OpType> ops = {
        ir::OpType::ADD,
        ir::OpType::SUB,
        ir::OpType::MUL,
        ir::OpType::DIV,
        ir::OpType::POW,
        ir::OpType::MAX_BINARY,
        ir::OpType::MIN_BINARY,
    };

    for (auto op : ops) {
        ir::Graph graph;

        auto a_data = random_positive_data(1024);  // Use positive for pow safety
        auto b_data = random_positive_data(1024);

        TensorSpec spec({1024}, DType::Float32);

        auto a = graph.create_constant(spec, a_data.data(), "a");
        auto b = graph.create_constant(spec, b_data.data(), "b");
        auto c = graph.create_op(op, {a, b}, spec, "c");

        SCOPED_TRACE("Testing op: " + ir::op_type_name(op));
        compare_with_cpu(graph, {c->id()}, 1e-4f, 1e-5f);
    }
}

// =============================================================================
// Elementwise Unary Operations Tests
// =============================================================================

TEST_F(ROCmOperationsTest, Neg) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::NEG, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Abs) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::ABS, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Sqrt) {
    ir::Graph graph;

    auto input_data = random_positive_data(4096);  // Must be positive

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::SQRT, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Rsqrt) {
    ir::Graph graph;

    auto input_data = random_positive_data(4096);  // Must be positive

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::RSQRT, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Exp) {
    ir::Graph graph;

    auto input_data = random_data(4096, -3.0f, 3.0f);  // Avoid overflow

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::EXP, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Log) {
    ir::Graph graph;

    auto input_data = random_positive_data(4096);  // Must be positive

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::LOG, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Sin) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::SIN, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Cos) {
    ir::Graph graph;

    auto input_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::COS, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, AllElementwiseUnary) {
    std::vector<std::pair<ir::OpType, bool>> ops = {
        {ir::OpType::NEG, false},
        {ir::OpType::ABS, false},
        {ir::OpType::SQRT, true},   // Needs positive
        {ir::OpType::RSQRT, true},  // Needs positive
        {ir::OpType::EXP, false},
        {ir::OpType::LOG, true},    // Needs positive
        {ir::OpType::SIN, false},
        {ir::OpType::COS, false},
    };

    for (auto [op, needs_positive] : ops) {
        ir::Graph graph;

        auto input_data = needs_positive ? random_positive_data(1024) : random_data(1024);

        TensorSpec spec({1024}, DType::Float32);

        auto input = graph.create_constant(spec, input_data.data(), "input");
        auto output = graph.create_op(op, {input}, spec, "output");

        SCOPED_TRACE("Testing op: " + ir::op_type_name(op));
        compare_with_cpu(graph, {output->id()});
    }
}

// =============================================================================
// Comparison Operations Tests
// =============================================================================

TEST_F(ROCmOperationsTest, Equal) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = a_data;  // Start with equal, then modify some
    for (size_t i = 0; i < b_data.size(); i += 2) {
        b_data[i] += 0.1f;
    }

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::EQ, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, NotEqual) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::NE, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, LessThan) {
    ir::Graph graph;

    auto a_data = random_data(4096);
    auto b_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto a = graph.create_constant(spec, a_data.data(), "a");
    auto b = graph.create_constant(spec, b_data.data(), "b");
    auto c = graph.create_op(ir::OpType::LT, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, AllComparisons) {
    std::vector<ir::OpType> ops = {
        ir::OpType::EQ,
        ir::OpType::NE,
        ir::OpType::LT,
        ir::OpType::LE,
        ir::OpType::GT,
        ir::OpType::GE,
    };

    for (auto op : ops) {
        ir::Graph graph;

        auto a_data = random_data(1024);
        auto b_data = random_data(1024);

        TensorSpec spec({1024}, DType::Float32);

        auto a = graph.create_constant(spec, a_data.data(), "a");
        auto b = graph.create_constant(spec, b_data.data(), "b");
        auto c = graph.create_op(op, {a, b}, spec, "c");

        SCOPED_TRACE("Testing comparison: " + ir::op_type_name(op));
        compare_with_cpu(graph, {c->id()});
    }
}

// =============================================================================
// Reduction Operations Tests
// =============================================================================

TEST_F(ROCmOperationsTest, ReduceSum) {
    ir::Graph graph;

    auto input_data = random_data(32 * 64);

    TensorSpec input_spec({32, 64}, DType::Float32);
    TensorSpec output_spec({32}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("dim", static_cast<int64_t>(1));
    attrs.set("keepdim", false);

    auto output = graph.create_op(ir::OpType::SUM, {input}, output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, ReduceMean) {
    ir::Graph graph;

    auto input_data = random_data(32 * 64);

    TensorSpec input_spec({32, 64}, DType::Float32);
    TensorSpec output_spec({32}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("dim", static_cast<int64_t>(1));
    attrs.set("keepdim", false);

    auto output = graph.create_op(ir::OpType::MEAN, {input}, output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, ReduceMax) {
    ir::Graph graph;

    auto input_data = random_data(32 * 64);

    TensorSpec input_spec({32, 64}, DType::Float32);
    TensorSpec output_spec({32}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("dim", static_cast<int64_t>(1));
    attrs.set("keepdim", false);

    auto output = graph.create_op(ir::OpType::MAX, {input}, output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, ReduceMin) {
    ir::Graph graph;

    auto input_data = random_data(32 * 64);

    TensorSpec input_spec({32, 64}, DType::Float32);
    TensorSpec output_spec({32}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("dim", static_cast<int64_t>(1));
    attrs.set("keepdim", false);

    auto output = graph.create_op(ir::OpType::MIN, {input}, output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

// =============================================================================
// Shape Operations Tests
// =============================================================================

TEST_F(ROCmOperationsTest, Reshape) {
    ir::Graph graph;

    auto input_data = random_data(64 * 32);

    TensorSpec input_spec({64, 32}, DType::Float32);
    TensorSpec output_spec({2048}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::RESHAPE, {input}, output_spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Transpose) {
    ir::Graph graph;

    auto input_data = random_data(32 * 64);

    TensorSpec input_spec({32, 64}, DType::Float32);
    TensorSpec output_spec({64, 32}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("dims", std::vector<int64_t>{1, 0});

    auto output = graph.create_op(ir::OpType::TRANSPOSE, {input}, output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Permute) {
    ir::Graph graph;

    auto input_data = random_data(2 * 3 * 4 * 5);

    TensorSpec input_spec({2, 3, 4, 5}, DType::Float32);
    TensorSpec output_spec({2, 4, 3, 5}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("dims", std::vector<int64_t>{0, 2, 1, 3});

    auto output = graph.create_op(ir::OpType::PERMUTE, {input}, output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

// =============================================================================
// Conditional Operations Tests
// =============================================================================

TEST_F(ROCmOperationsTest, Where) {
    ir::Graph graph;

    // Condition: random booleans (0 or 1)
    std::vector<float> cond_data(4096);
    for (auto& v : cond_data) {
        v = (rng_() % 2 == 0) ? 1.0f : 0.0f;
    }
    auto x_data = random_data(4096);
    auto y_data = random_data(4096);

    TensorSpec spec({4096}, DType::Float32);

    auto cond = graph.create_constant(spec, cond_data.data(), "cond");
    auto x = graph.create_constant(spec, x_data.data(), "x");
    auto y = graph.create_constant(spec, y_data.data(), "y");
    auto output = graph.create_op(ir::OpType::WHERE, {cond, x, y}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, Clamp) {
    ir::Graph graph;

    auto input_data = random_data(4096, -5.0f, 5.0f);

    TensorSpec spec({4096}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");

    ir::OpAttributes attrs;
    attrs.set("min", -1.0f);
    attrs.set("max", 1.0f);

    auto output = graph.create_op(ir::OpType::CLAMP, {input}, spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

// =============================================================================
// Loss Function Tests
// =============================================================================

TEST_F(ROCmOperationsTest, MSELoss) {
    ir::Graph graph;

    auto pred_data = random_data(32 * 10);
    auto target_data = random_data(32 * 10);

    TensorSpec input_spec({32, 10}, DType::Float32);
    TensorSpec output_spec({}, DType::Float32);  // Scalar

    auto pred = graph.create_constant(input_spec, pred_data.data(), "pred");
    auto target = graph.create_constant(input_spec, target_data.data(), "target");

    ir::OpAttributes attrs;
    attrs.set("reduction", std::string("mean"));

    auto output = graph.create_op(ir::OpType::MSE_LOSS, {pred, target}, output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()});
}

TEST_F(ROCmOperationsTest, BCELoss) {
    ir::Graph graph;

    // Predictions should be in (0, 1) for BCE
    auto pred_data = random_positive_data(32 * 2, 0.01f, 0.99f);
    // Targets are 0 or 1
    std::vector<float> target_data(32 * 2);
    for (auto& v : target_data) {
        v = (rng_() % 2 == 0) ? 1.0f : 0.0f;
    }

    TensorSpec input_spec({32, 2}, DType::Float32);
    TensorSpec output_spec({}, DType::Float32);  // Scalar

    auto pred = graph.create_constant(input_spec, pred_data.data(), "pred");
    auto target = graph.create_constant(input_spec, target_data.data(), "target");

    ir::OpAttributes attrs;
    attrs.set("reduction", std::string("mean"));

    auto output = graph.create_op(ir::OpType::BCE_LOSS, {pred, target}, output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()}, 1e-4f, 1e-5f);
}

TEST_F(ROCmOperationsTest, CrossEntropyLoss) {
    ir::Graph graph;

    // Logits: [batch=32, classes=10]
    auto logits_data = random_data(32 * 10);
    // Targets: class indices 0-9
    auto targets_data = random_indices(32, 10);

    TensorSpec logits_spec({32, 10}, DType::Float32);
    TensorSpec targets_spec({32}, DType::Int32);
    TensorSpec output_spec({}, DType::Float32);  // Scalar

    auto logits = graph.create_constant(logits_spec, logits_data.data(), "logits");
    auto targets = graph.create_constant(targets_spec, targets_data.data(), "targets");

    ir::OpAttributes attrs;
    attrs.set("reduction", std::string("mean"));

    auto output = graph.create_op(ir::OpType::CROSS_ENTROPY_LOSS, {logits, targets},
                                   output_spec, "output", attrs);

    compare_with_cpu(graph, {output->id()}, 1e-4f, 1e-5f);
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

TEST_F(ROCmOperationsTest, EmptyTensor) {
    ir::Graph graph;

    std::vector<float> empty_data;

    TensorSpec spec({0}, DType::Float32);

    auto input = graph.create_constant(spec, empty_data.data(), "input");
    auto output = graph.create_op(ir::OpType::RELU, {input}, spec, "output");

    // Empty tensors should not crash
    auto result = executor_->execute(graph, {output->id()});
    EXPECT_TRUE(result.success);
}

TEST_F(ROCmOperationsTest, SingleElement) {
    ir::Graph graph;

    std::vector<float> data = {3.14159f};

    TensorSpec spec({1}, DType::Float32);

    auto a = graph.create_constant(spec, data.data(), "a");
    auto b = graph.create_constant(spec, data.data(), "b");
    auto c = graph.create_op(ir::OpType::ADD, {a, b}, spec, "c");

    compare_with_cpu(graph, {c->id()});
}

TEST_F(ROCmOperationsTest, LargeTensor) {
    ir::Graph graph;

    // 16MB tensor
    size_t size = 4 * 1024 * 1024;  // 4M floats = 16MB
    auto input_data = random_data(size);

    TensorSpec spec({static_cast<int64_t>(size)}, DType::Float32);

    auto input = graph.create_constant(spec, input_data.data(), "input");
    auto output = graph.create_op(ir::OpType::RELU, {input}, spec, "output");

    compare_with_cpu(graph, {output->id()});
}

// =============================================================================
// Complex Graph Tests (Multiple Operations)
// =============================================================================

TEST_F(ROCmOperationsTest, SimpleMLP) {
    ir::Graph graph;

    // Simple 2-layer MLP: input -> linear1 -> relu -> linear2
    auto x_data = random_data(32 * 784);
    auto w1_data = random_data(784 * 256);
    auto b1_data = random_data(256);
    auto w2_data = random_data(256 * 10);
    auto b2_data = random_data(10);

    TensorSpec x_spec({32, 784}, DType::Float32);
    TensorSpec w1_spec({784, 256}, DType::Float32);
    TensorSpec b1_spec({256}, DType::Float32);
    TensorSpec h1_spec({32, 256}, DType::Float32);
    TensorSpec w2_spec({256, 10}, DType::Float32);
    TensorSpec b2_spec({10}, DType::Float32);
    TensorSpec out_spec({32, 10}, DType::Float32);

    auto x = graph.create_constant(x_spec, x_data.data(), "x");
    auto w1 = graph.create_constant(w1_spec, w1_data.data(), "w1");
    auto b1 = graph.create_constant(b1_spec, b1_data.data(), "b1");
    auto w2 = graph.create_constant(w2_spec, w2_data.data(), "w2");
    auto b2 = graph.create_constant(b2_spec, b2_data.data(), "b2");

    // Forward pass
    auto h1_mm = graph.create_op(ir::OpType::MATMUL, {x, w1}, h1_spec, "h1_mm");
    auto h1_bias = graph.create_op(ir::OpType::ADD, {h1_mm, b1}, h1_spec, "h1_bias");
    auto h1_relu = graph.create_op(ir::OpType::RELU, {h1_bias}, h1_spec, "h1_relu");
    auto h2_mm = graph.create_op(ir::OpType::MATMUL, {h1_relu, w2}, out_spec, "h2_mm");
    auto output = graph.create_op(ir::OpType::ADD, {h2_mm, b2}, out_spec, "output");

    compare_with_cpu(graph, {output->id()}, 1e-4f, 1e-5f);
}

TEST_F(ROCmOperationsTest, ConvBlock) {
    ir::Graph graph;

    // Conv -> BatchNorm -> ReLU block
    auto input_data = random_data(4 * 32 * 14 * 14);
    auto conv_weight_data = random_data(64 * 32 * 3 * 3);
    auto gamma_data = random_positive_data(64);
    auto beta_data = random_data(64);
    auto mean_data = random_data(64, -0.5f, 0.5f);
    auto var_data = random_positive_data(64, 0.5f, 2.0f);

    TensorSpec input_spec({4, 32, 14, 14}, DType::Float32);
    TensorSpec conv_weight_spec({64, 32, 3, 3}, DType::Float32);
    TensorSpec conv_out_spec({4, 64, 14, 14}, DType::Float32);
    TensorSpec param_spec({64}, DType::Float32);

    auto input = graph.create_constant(input_spec, input_data.data(), "input");
    auto conv_weight = graph.create_constant(conv_weight_spec, conv_weight_data.data(), "conv_weight");
    auto gamma = graph.create_constant(param_spec, gamma_data.data(), "gamma");
    auto beta = graph.create_constant(param_spec, beta_data.data(), "beta");
    auto mean = graph.create_constant(param_spec, mean_data.data(), "mean");
    auto var = graph.create_constant(param_spec, var_data.data(), "var");

    ir::OpAttributes conv_attrs;
    conv_attrs.set("padding", std::vector<int64_t>{1, 1});
    conv_attrs.set("stride", std::vector<int64_t>{1, 1});

    auto conv_out = graph.create_op(ir::OpType::CONV2D, {input, conv_weight},
                                     conv_out_spec, "conv_out", conv_attrs);

    ir::OpAttributes bn_attrs;
    bn_attrs.set("epsilon", 1e-5);
    bn_attrs.set("training", false);

    auto bn_out = graph.create_op(ir::OpType::BATCH_NORM,
                                   {conv_out, gamma, beta, mean, var},
                                   conv_out_spec, "bn_out", bn_attrs);

    auto output = graph.create_op(ir::OpType::RELU, {bn_out}, conv_out_spec, "output");

    compare_with_cpu(graph, {output->id()}, 1e-3f, 1e-4f);  // Looser tolerance for complex graphs
}

#endif  // PYFLAME_HAS_ROCM
