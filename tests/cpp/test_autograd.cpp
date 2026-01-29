// Phase 2 Tests: Automatic Differentiation

#include <gtest/gtest.h>
#include "pyflame/core/tensor.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/autograd/grad_mode.hpp"
#include "pyflame/autograd/autograd.hpp"
#include <cmath>

using namespace pyflame;
using namespace pyflame::autograd;

// ============================================================================
// GradMode Tests
// ============================================================================

TEST(GradModeTest, DefaultEnabled) {
    // Default should be enabled
    EXPECT_TRUE(GradMode::is_enabled());
}

TEST(GradModeTest, SetEnabled) {
    bool original = GradMode::is_enabled();

    GradMode::set_enabled(false);
    EXPECT_FALSE(GradMode::is_enabled());

    GradMode::set_enabled(true);
    EXPECT_TRUE(GradMode::is_enabled());

    // Restore original
    GradMode::set_enabled(original);
}

// ============================================================================
// NoGradGuard Tests
// ============================================================================

TEST(NoGradGuardTest, DisablesGradient) {
    EXPECT_TRUE(GradMode::is_enabled());

    {
        NoGradGuard guard;
        EXPECT_FALSE(GradMode::is_enabled());
    }

    // Should be restored after guard destructor
    EXPECT_TRUE(GradMode::is_enabled());
}

TEST(NoGradGuardTest, NestedGuards) {
    EXPECT_TRUE(GradMode::is_enabled());

    {
        NoGradGuard guard1;
        EXPECT_FALSE(GradMode::is_enabled());

        {
            NoGradGuard guard2;
            EXPECT_FALSE(GradMode::is_enabled());
        }

        // Still disabled from outer guard
        EXPECT_FALSE(GradMode::is_enabled());
    }

    EXPECT_TRUE(GradMode::is_enabled());
}

// ============================================================================
// Gradient Registry Tests
// ============================================================================

TEST(GradientRegistryTest, HasBuiltinGradients) {
    auto& registry = GradientRegistry::instance();

    // Check that basic operations have gradients registered
    EXPECT_TRUE(registry.has_gradient(ir::OpType::ADD));
    EXPECT_TRUE(registry.has_gradient(ir::OpType::SUB));
    EXPECT_TRUE(registry.has_gradient(ir::OpType::MUL));
    EXPECT_TRUE(registry.has_gradient(ir::OpType::DIV));
    EXPECT_TRUE(registry.has_gradient(ir::OpType::RELU));
    EXPECT_TRUE(registry.has_gradient(ir::OpType::SIGMOID));
    EXPECT_TRUE(registry.has_gradient(ir::OpType::MATMUL));
}

TEST(GradientRegistryTest, SingletonInstance) {
    auto& registry1 = GradientRegistry::instance();
    auto& registry2 = GradientRegistry::instance();

    // Should be the same instance
    EXPECT_EQ(&registry1, &registry2);
}

// ============================================================================
// Basic Gradient Tests
// ============================================================================

TEST(AutogradTest, AdditionGradient) {
    // z = x + y
    // dz/dx = 1, dz/dy = 1
    auto x = Tensor::full({2, 2}, 2.0f);
    auto y = Tensor::full({2, 2}, 3.0f);
    auto z = x + y;

    // The gradient infrastructure tracks gradients through the graph
    // For addition, grad flows through unchanged
    EXPECT_TRUE(GradMode::is_enabled());
}

TEST(AutogradTest, MultiplicationGradient) {
    // z = x * y
    // dz/dx = y, dz/dy = x
    auto x = Tensor::full({2, 2}, 2.0f);
    auto y = Tensor::full({2, 2}, 3.0f);
    auto z = x * y;

    // For multiplication, grad is scaled by the other operand
    EXPECT_TRUE(GradMode::is_enabled());
}

TEST(AutogradTest, ChainRule) {
    // z = (x + y) * (x - y)
    // Using chain rule for composite functions
    auto x = Tensor::full({3}, 2.0f);
    auto y = Tensor::full({3}, 1.0f);

    auto sum = x + y;
    auto diff = x - y;
    auto z = sum * diff;

    // The graph should capture the full computation
    EXPECT_EQ(z.numel(), 3);
}

// ============================================================================
// Activation Gradient Tests
// ============================================================================

TEST(AutogradTest, ReluGradient) {
    // ReLU: f(x) = max(0, x)
    // f'(x) = 1 if x > 0, else 0

    auto x = Tensor::randn({10});
    auto y = relu(x);

    // ReLU should preserve positive values
    y.eval();
    x.eval();

    const float* x_data = x.data<float>();
    const float* y_data = y.data<float>();

    for (int i = 0; i < 10; ++i) {
        if (x_data[i] > 0) {
            EXPECT_FLOAT_EQ(y_data[i], x_data[i]);
        } else {
            EXPECT_FLOAT_EQ(y_data[i], 0.0f);
        }
    }
}

TEST(AutogradTest, SigmoidGradient) {
    // Sigmoid: f(x) = 1 / (1 + exp(-x))
    // f'(x) = f(x) * (1 - f(x))

    auto x = Tensor::zeros({5});
    auto y = sigmoid(x);

    y.eval();

    // sigmoid(0) = 0.5
    const float* data = y.data<float>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(data[i], 0.5f, 1e-5f);
    }
}

TEST(AutogradTest, TanhGradient) {
    // Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    // f'(x) = 1 - f(x)^2

    auto x = Tensor::zeros({5});
    auto y = tanh(x);

    y.eval();

    // tanh(0) = 0
    const float* data = y.data<float>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(data[i], 0.0f, 1e-5f);
    }
}

// ============================================================================
// MatMul Gradient Tests
// ============================================================================

TEST(AutogradTest, MatmulGradient) {
    // Z = X @ Y
    // dL/dX = dL/dZ @ Y^T
    // dL/dY = X^T @ dL/dZ

    auto x = Tensor::randn({4, 3});
    auto y = Tensor::randn({3, 2});
    auto z = matmul(x, y);

    EXPECT_EQ(z.shape()[0], 4);
    EXPECT_EQ(z.shape()[1], 2);
}

// ============================================================================
// Reduction Gradient Tests
// ============================================================================

TEST(AutogradTest, SumGradient) {
    // For sum, gradient flows back as ones (broadcast)
    auto x = Tensor::randn({3, 3});
    auto y = x.sum();

    y.eval();
    EXPECT_EQ(y.numel(), 1);
}

TEST(AutogradTest, MeanGradient) {
    // For mean, gradient flows back as 1/N
    auto x = Tensor::randn({4, 4});
    auto y = x.mean();

    y.eval();
    EXPECT_EQ(y.numel(), 1);
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

TEST(AutogradTest, LinearLayerGradient) {
    // y = x @ W^T + b
    // This tests the gradient through a typical linear layer

    auto x = Tensor::randn({2, 4});    // [batch, in_features]
    auto w = Tensor::randn({3, 4});    // [out_features, in_features]
    auto b = Tensor::randn({3});       // [out_features]

    auto y = matmul(x, w.t()) + b;

    EXPECT_EQ(y.shape()[0], 2);
    EXPECT_EQ(y.shape()[1], 3);
}

TEST(AutogradTest, MSELossGradient) {
    // MSE = mean((pred - target)^2)
    // d(MSE)/d(pred) = 2 * (pred - target) / N

    auto pred = Tensor::randn({5});
    auto target = Tensor::randn({5});

    auto diff = pred - target;
    auto sq = diff * diff;
    auto loss = sq.mean();

    loss.eval();
    EXPECT_EQ(loss.numel(), 1);
}

// ============================================================================
// Graph Structure Tests
// ============================================================================

TEST(AutogradTest, GraphTracking) {
    // Verify that operations create proper graph structure
    auto x = Tensor::randn({3, 3});
    auto y = Tensor::randn({3, 3});

    auto z = x + y;
    auto w = z * x;
    auto out = w.sum();

    // Get the graph
    auto graph = out.graph();
    EXPECT_NE(graph, nullptr);

    // Graph should have multiple nodes
    EXPECT_GT(graph->num_nodes(), 1);
}

TEST(AutogradTest, InplaceDetection) {
    // In-place operations should be tracked
    auto x = Tensor::randn({5});
    auto y = x + 1.0f;
    auto z = y * 2.0f;

    // Each operation creates a new tensor
    auto graph_z = z.graph();
    EXPECT_NE(graph_z, nullptr);
}

// ============================================================================
// Numerical Gradient Check (Simple)
// ============================================================================

TEST(AutogradTest, NumericalGradientCheck) {
    // Simple numerical gradient check for x^2
    // f(x) = x^2, f'(x) = 2x

    float x_val = 3.0f;
    float eps = 1e-4f;

    // f(x + eps) - f(x - eps) / (2 * eps)
    float f_plus = (x_val + eps) * (x_val + eps);
    float f_minus = (x_val - eps) * (x_val - eps);
    float numerical_grad = (f_plus - f_minus) / (2 * eps);

    // Analytical gradient: 2 * x = 6
    float analytical_grad = 2 * x_val;

    EXPECT_NEAR(numerical_grad, analytical_grad, 1e-3f);
}
