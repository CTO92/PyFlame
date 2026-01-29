/// @file test_rocm_kernels.cpp
/// @brief Tests for custom HIP kernels

#include <gtest/gtest.h>

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_kernels.hpp"
#include "pyflame/backend/rocm/rocm_memory.hpp"
#include "pyflame/backend/rocm/rocm_backend.hpp"
#include "pyflame/ir/op_type.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <random>

using namespace pyflame::backend::rocm;
using namespace pyflame::backend::rocm::kernels;

class ROCmKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!is_available()) {
            GTEST_SKIP() << "ROCm not available";
        }
        set_device(0);
        memory_ = std::make_unique<ROCmMemoryManager>();
        hipStreamCreate(&stream_);
    }

    void TearDown() override {
        if (stream_) {
            hipStreamDestroy(stream_);
        }
        memory_.reset();
    }

    // Allocate GPU memory and copy host data
    void* to_device(const std::vector<float>& host_data) {
        size_t size = host_data.size() * sizeof(float);
        void* ptr = memory_->allocate(size);
        memory_->copy_host_to_device(ptr, host_data.data(), size);
        return ptr;
    }

    // Copy GPU data back to host
    std::vector<float> to_host(void* device_ptr, size_t count) {
        std::vector<float> result(count);
        memory_->copy_device_to_host(result.data(), device_ptr, count * sizeof(float));
        return result;
    }

    // Generate random test data
    std::vector<float> random_data(size_t size, float min = -1.0f, float max = 1.0f) {
        std::vector<float> data(size);
        std::uniform_real_distribution<float> dist(min, max);
        for (auto& v : data) {
            v = dist(rng_);
        }
        return data;
    }

    // CPU reference implementations
    float gelu_cpu(float x) {
        return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    float silu_cpu(float x) {
        return x / (1.0f + std::exp(-x));
    }

    std::unique_ptr<ROCmMemoryManager> memory_;
    hipStream_t stream_ = nullptr;
    std::mt19937 rng_{42};
};

// =============================================================================
// GELU Kernel Tests
// =============================================================================

TEST_F(ROCmKernelsTest, GELU_Basic) {
    const size_t count = 1024;
    auto input = random_data(count);

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_gelu_forward(
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    // Compare with CPU reference
    for (size_t i = 0; i < count; ++i) {
        float expected = gelu_cpu(input[i]);
        EXPECT_NEAR(output[i], expected, 1e-4f)
            << "GELU mismatch at index " << i;
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

TEST_F(ROCmKernelsTest, GELU_Large) {
    const size_t count = 1024 * 1024;  // 1M elements
    auto input = random_data(count);

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_gelu_forward(
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    // Spot check a few values
    for (size_t i = 0; i < count; i += 10000) {
        float expected = gelu_cpu(input[i]);
        EXPECT_NEAR(output[i], expected, 1e-4f);
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

// =============================================================================
// SiLU Kernel Tests
// =============================================================================

TEST_F(ROCmKernelsTest, SiLU_Basic) {
    const size_t count = 1024;
    auto input = random_data(count);

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_silu_forward(
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    for (size_t i = 0; i < count; ++i) {
        float expected = silu_cpu(input[i]);
        EXPECT_NEAR(output[i], expected, 1e-5f)
            << "SiLU mismatch at index " << i;
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

// =============================================================================
// Elementwise Binary Kernel Tests
// =============================================================================

TEST_F(ROCmKernelsTest, ElementwiseAdd) {
    const size_t count = 4096;
    auto a = random_data(count);
    auto b = random_data(count);

    void* d_a = to_device(a);
    void* d_b = to_device(b);
    void* d_out = memory_->allocate(count * sizeof(float));

    launch_elementwise_binary(
        pyflame::ir::OpType::ADD,
        static_cast<float*>(d_out),
        static_cast<const float*>(d_a),
        static_cast<const float*>(d_b),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_out, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], a[i] + b[i], 1e-6f);
    }

    memory_->deallocate(d_a);
    memory_->deallocate(d_b);
    memory_->deallocate(d_out);
}

TEST_F(ROCmKernelsTest, ElementwiseSub) {
    const size_t count = 4096;
    auto a = random_data(count);
    auto b = random_data(count);

    void* d_a = to_device(a);
    void* d_b = to_device(b);
    void* d_out = memory_->allocate(count * sizeof(float));

    launch_elementwise_binary(
        pyflame::ir::OpType::SUB,
        static_cast<float*>(d_out),
        static_cast<const float*>(d_a),
        static_cast<const float*>(d_b),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_out, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], a[i] - b[i], 1e-6f);
    }

    memory_->deallocate(d_a);
    memory_->deallocate(d_b);
    memory_->deallocate(d_out);
}

TEST_F(ROCmKernelsTest, ElementwiseMul) {
    const size_t count = 4096;
    auto a = random_data(count);
    auto b = random_data(count);

    void* d_a = to_device(a);
    void* d_b = to_device(b);
    void* d_out = memory_->allocate(count * sizeof(float));

    launch_elementwise_binary(
        pyflame::ir::OpType::MUL,
        static_cast<float*>(d_out),
        static_cast<const float*>(d_a),
        static_cast<const float*>(d_b),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_out, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], a[i] * b[i], 1e-6f);
    }

    memory_->deallocate(d_a);
    memory_->deallocate(d_b);
    memory_->deallocate(d_out);
}

TEST_F(ROCmKernelsTest, ElementwiseDiv) {
    const size_t count = 4096;
    auto a = random_data(count);
    auto b = random_data(count, 0.5f, 2.0f);  // Avoid division by near-zero

    void* d_a = to_device(a);
    void* d_b = to_device(b);
    void* d_out = memory_->allocate(count * sizeof(float));

    launch_elementwise_binary(
        pyflame::ir::OpType::DIV,
        static_cast<float*>(d_out),
        static_cast<const float*>(d_a),
        static_cast<const float*>(d_b),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_out, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], a[i] / b[i], 1e-5f);
    }

    memory_->deallocate(d_a);
    memory_->deallocate(d_b);
    memory_->deallocate(d_out);
}

// =============================================================================
// Elementwise Unary Kernel Tests
// =============================================================================

TEST_F(ROCmKernelsTest, ElementwiseNeg) {
    const size_t count = 4096;
    auto input = random_data(count);

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_elementwise_unary(
        pyflame::ir::OpType::NEG,
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], -input[i], 1e-6f);
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

TEST_F(ROCmKernelsTest, ElementwiseAbs) {
    const size_t count = 4096;
    auto input = random_data(count);

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_elementwise_unary(
        pyflame::ir::OpType::ABS,
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], std::abs(input[i]), 1e-6f);
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

TEST_F(ROCmKernelsTest, ElementwiseSqrt) {
    const size_t count = 4096;
    auto input = random_data(count, 0.1f, 10.0f);  // Positive values only

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_elementwise_unary(
        pyflame::ir::OpType::SQRT,
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], std::sqrt(input[i]), 1e-5f);
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

TEST_F(ROCmKernelsTest, ElementwiseExp) {
    const size_t count = 4096;
    auto input = random_data(count, -3.0f, 3.0f);  // Avoid overflow

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_elementwise_unary(
        pyflame::ir::OpType::EXP,
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], std::exp(input[i]), 1e-4f);
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

TEST_F(ROCmKernelsTest, ElementwiseLog) {
    const size_t count = 4096;
    auto input = random_data(count, 0.1f, 10.0f);  // Positive values

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_elementwise_unary(
        pyflame::ir::OpType::LOG,
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(output[i], std::log(input[i]), 1e-5f);
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

// =============================================================================
// Comparison Kernel Tests
// =============================================================================

TEST_F(ROCmKernelsTest, ComparisonEQ) {
    const size_t count = 4096;
    auto a = random_data(count);
    auto b = a;  // Start equal
    // Modify some elements
    for (size_t i = 0; i < count; i += 2) {
        b[i] += 0.1f;
    }

    void* d_a = to_device(a);
    void* d_b = to_device(b);
    void* d_out = memory_->allocate(count * sizeof(float));

    launch_comparison(
        pyflame::ir::OpType::EQ,
        static_cast<float*>(d_out),
        static_cast<const float*>(d_a),
        static_cast<const float*>(d_b),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_out, count);

    for (size_t i = 0; i < count; ++i) {
        float expected = (a[i] == b[i]) ? 1.0f : 0.0f;
        EXPECT_EQ(output[i], expected) << "EQ mismatch at index " << i;
    }

    memory_->deallocate(d_a);
    memory_->deallocate(d_b);
    memory_->deallocate(d_out);
}

TEST_F(ROCmKernelsTest, ComparisonLT) {
    const size_t count = 4096;
    auto a = random_data(count);
    auto b = random_data(count);

    void* d_a = to_device(a);
    void* d_b = to_device(b);
    void* d_out = memory_->allocate(count * sizeof(float));

    launch_comparison(
        pyflame::ir::OpType::LT,
        static_cast<float*>(d_out),
        static_cast<const float*>(d_a),
        static_cast<const float*>(d_b),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_out, count);

    for (size_t i = 0; i < count; ++i) {
        float expected = (a[i] < b[i]) ? 1.0f : 0.0f;
        EXPECT_EQ(output[i], expected) << "LT mismatch at index " << i;
    }

    memory_->deallocate(d_a);
    memory_->deallocate(d_b);
    memory_->deallocate(d_out);
}

// =============================================================================
// Conditional Kernel Tests
// =============================================================================

TEST_F(ROCmKernelsTest, Where) {
    const size_t count = 4096;

    // Create condition (0 or 1)
    std::vector<float> cond(count);
    for (size_t i = 0; i < count; ++i) {
        cond[i] = (i % 2 == 0) ? 1.0f : 0.0f;
    }
    auto x = random_data(count);
    auto y = random_data(count);

    void* d_cond = to_device(cond);
    void* d_x = to_device(x);
    void* d_y = to_device(y);
    void* d_out = memory_->allocate(count * sizeof(float));

    launch_where(
        static_cast<float*>(d_out),
        static_cast<const float*>(d_cond),
        static_cast<const float*>(d_x),
        static_cast<const float*>(d_y),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_out, count);

    for (size_t i = 0; i < count; ++i) {
        float expected = (cond[i] != 0.0f) ? x[i] : y[i];
        EXPECT_EQ(output[i], expected) << "Where mismatch at index " << i;
    }

    memory_->deallocate(d_cond);
    memory_->deallocate(d_x);
    memory_->deallocate(d_y);
    memory_->deallocate(d_out);
}

TEST_F(ROCmKernelsTest, Clamp) {
    const size_t count = 4096;
    auto input = random_data(count, -5.0f, 5.0f);
    float min_val = -1.0f;
    float max_val = 1.0f;

    void* d_input = to_device(input);
    void* d_output = memory_->allocate(count * sizeof(float));

    launch_clamp(
        static_cast<float*>(d_output),
        static_cast<const float*>(d_input),
        min_val,
        max_val,
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, count);

    for (size_t i = 0; i < count; ++i) {
        float expected = std::max(min_val, std::min(max_val, input[i]));
        EXPECT_NEAR(output[i], expected, 1e-6f) << "Clamp mismatch at index " << i;
    }

    memory_->deallocate(d_input);
    memory_->deallocate(d_output);
}

// =============================================================================
// Loss Function Kernel Tests
// =============================================================================

TEST_F(ROCmKernelsTest, MSELoss) {
    const size_t count = 1024;
    auto pred = random_data(count);
    auto target = random_data(count);

    // CPU reference
    float expected = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float diff = pred[i] - target[i];
        expected += diff * diff;
    }
    expected /= count;

    void* d_pred = to_device(pred);
    void* d_target = to_device(target);
    void* d_output = memory_->allocate(sizeof(float));

    launch_mse_loss(
        static_cast<float*>(d_output),
        static_cast<const float*>(d_pred),
        static_cast<const float*>(d_target),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, 1);

    EXPECT_NEAR(output[0], expected, 1e-4f) << "MSE loss mismatch";

    memory_->deallocate(d_pred);
    memory_->deallocate(d_target);
    memory_->deallocate(d_output);
}

TEST_F(ROCmKernelsTest, BCELoss) {
    const size_t count = 1024;
    auto pred = random_data(count, 0.01f, 0.99f);  // Valid probability range
    std::vector<float> target(count);
    for (size_t i = 0; i < count; ++i) {
        target[i] = (i % 2 == 0) ? 1.0f : 0.0f;
    }

    // CPU reference
    float expected = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        expected -= target[i] * std::log(pred[i] + 1e-7f)
                  + (1.0f - target[i]) * std::log(1.0f - pred[i] + 1e-7f);
    }
    expected /= count;

    void* d_pred = to_device(pred);
    void* d_target = to_device(target);
    void* d_output = memory_->allocate(sizeof(float));

    launch_bce_loss(
        static_cast<float*>(d_output),
        static_cast<const float*>(d_pred),
        static_cast<const float*>(d_target),
        count,
        stream_
    );
    hipStreamSynchronize(stream_);

    auto output = to_host(d_output, 1);

    EXPECT_NEAR(output[0], expected, 1e-3f) << "BCE loss mismatch";

    memory_->deallocate(d_pred);
    memory_->deallocate(d_target);
    memory_->deallocate(d_output);
}

#endif  // PYFLAME_HAS_ROCM
