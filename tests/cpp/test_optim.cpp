// Phase 2 Tests: Optimizers and Learning Rate Schedulers

#include <gtest/gtest.h>
#include "pyflame/core/tensor.hpp"
#include "pyflame/nn/linear.hpp"
#include "pyflame/optim/optimizer.hpp"
#include "pyflame/optim/lr_scheduler.hpp"
#include <cmath>

using namespace pyflame;
using namespace pyflame::nn;
using namespace pyflame::optim;

// Helper to create a simple model with known parameters
std::shared_ptr<Linear> create_test_model() {
    return std::make_shared<Linear>(4, 2, false);
}

// ============================================================================
// SGD Tests
// ============================================================================

TEST(SGDTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();

    SGD optimizer(params, 0.1f);

    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.1f);
    EXPECT_FLOAT_EQ(optimizer.momentum(), 0.0f);
    EXPECT_FLOAT_EQ(optimizer.weight_decay(), 0.0f);
}

TEST(SGDTest, ConstructionWithMomentum) {
    auto model = create_test_model();
    auto params = model->parameters();

    SGD optimizer(params, 0.01f, 0.9f);

    EXPECT_FLOAT_EQ(optimizer.momentum(), 0.9f);
}

TEST(SGDTest, SetLearningRate) {
    auto model = create_test_model();
    auto params = model->parameters();

    SGD optimizer(params, 0.1f);
    optimizer.set_lr(0.01f);

    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.01f);
}

TEST(SGDTest, ZeroGrad) {
    auto model = create_test_model();
    auto params = model->parameters();

    SGD optimizer(params, 0.1f);
    optimizer.zero_grad();

    // Should not throw
    SUCCEED();
}

TEST(SGDTest, ToString) {
    auto model = create_test_model();
    auto params = model->parameters();

    SGD optimizer(params, 0.1f, 0.9f, 0.0f, 0.001f, true);
    std::string s = optimizer.to_string();

    EXPECT_TRUE(s.find("SGD") != std::string::npos);
    EXPECT_TRUE(s.find("momentum") != std::string::npos);
    EXPECT_TRUE(s.find("nesterov") != std::string::npos);
}

TEST(SGDTest, StateDict) {
    auto model = create_test_model();
    auto params = model->parameters();

    SGD optimizer(params, 0.1f, 0.9f);
    optimizer.step();  // Run one step to create momentum buffers

    auto state = optimizer.state_dict();
    EXPECT_TRUE(state.count("step") > 0);
}

// ============================================================================
// Adam Tests
// ============================================================================

TEST(AdamTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();

    Adam optimizer(params, 0.001f);

    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.001f);
    EXPECT_FLOAT_EQ(optimizer.beta1(), 0.9f);
    EXPECT_FLOAT_EQ(optimizer.beta2(), 0.999f);
}

TEST(AdamTest, CustomBetas) {
    auto model = create_test_model();
    auto params = model->parameters();

    Adam optimizer(params, 0.001f, 0.8f, 0.99f);

    EXPECT_FLOAT_EQ(optimizer.beta1(), 0.8f);
    EXPECT_FLOAT_EQ(optimizer.beta2(), 0.99f);
}

TEST(AdamTest, AMSGrad) {
    auto model = create_test_model();
    auto params = model->parameters();

    Adam optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, true);

    EXPECT_TRUE(optimizer.amsgrad());
}

TEST(AdamTest, ToString) {
    auto model = create_test_model();
    auto params = model->parameters();

    Adam optimizer(params);
    std::string s = optimizer.to_string();

    EXPECT_TRUE(s.find("Adam") != std::string::npos);
    EXPECT_TRUE(s.find("betas") != std::string::npos);
}

// ============================================================================
// AdamW Tests
// ============================================================================

TEST(AdamWTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();

    AdamW optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

    EXPECT_FLOAT_EQ(optimizer.weight_decay(), 0.01f);
}

TEST(AdamWTest, ToString) {
    auto model = create_test_model();
    auto params = model->parameters();

    AdamW optimizer(params);
    std::string s = optimizer.to_string();

    EXPECT_TRUE(s.find("AdamW") != std::string::npos);
    EXPECT_TRUE(s.find("weight_decay") != std::string::npos);
}

// ============================================================================
// RMSprop Tests
// ============================================================================

TEST(RMSpropTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();

    RMSprop optimizer(params, 0.01f);

    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.01f);
    EXPECT_FLOAT_EQ(optimizer.alpha(), 0.99f);
}

TEST(RMSpropTest, Centered) {
    auto model = create_test_model();
    auto params = model->parameters();

    RMSprop optimizer(params, 0.01f, 0.99f, 1e-8f, 0.0f, 0.0f, true);

    EXPECT_TRUE(optimizer.centered());
}

TEST(RMSpropTest, ToString) {
    auto model = create_test_model();
    auto params = model->parameters();

    RMSprop optimizer(params);
    std::string s = optimizer.to_string();

    EXPECT_TRUE(s.find("RMSprop") != std::string::npos);
}

// ============================================================================
// StepLR Tests
// ============================================================================

TEST(StepLRTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 0.1f);

    StepLR scheduler(optimizer, 10, 0.1f);

    EXPECT_EQ(scheduler.step_size(), 10);
    EXPECT_FLOAT_EQ(scheduler.gamma(), 0.1f);
}

TEST(StepLRTest, Step) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 1.0f);

    StepLR scheduler(optimizer, 5, 0.5f);

    // Initial LR = 1.0
    EXPECT_FLOAT_EQ(scheduler.get_lr(), 1.0f);

    // After 5 steps, LR should decay
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    // LR should be 1.0 * 0.5 = 0.5
    EXPECT_NEAR(scheduler.get_lr(), 0.5f, 1e-5f);

    // After 5 more steps
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    // LR should be 1.0 * 0.5^2 = 0.25
    EXPECT_NEAR(scheduler.get_lr(), 0.25f, 1e-5f);
}

TEST(StepLRTest, ToString) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 0.1f);

    StepLR scheduler(optimizer, 10);
    std::string s = scheduler.to_string();

    EXPECT_TRUE(s.find("StepLR") != std::string::npos);
}

// ============================================================================
// MultiStepLR Tests
// ============================================================================

TEST(MultiStepLRTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 0.1f);

    MultiStepLR scheduler(optimizer, {30, 80}, 0.1f);

    auto milestones = scheduler.milestones();
    EXPECT_EQ(milestones.size(), 2);
    EXPECT_EQ(milestones[0], 30);
    EXPECT_EQ(milestones[1], 80);
}

TEST(MultiStepLRTest, Step) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 1.0f);

    MultiStepLR scheduler(optimizer, {5, 10}, 0.5f);

    // Before first milestone
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_lr(), 0.5f, 1e-5f);

    // After second milestone
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_lr(), 0.25f, 1e-5f);
}

// ============================================================================
// ExponentialLR Tests
// ============================================================================

TEST(ExponentialLRTest, Step) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 1.0f);

    ExponentialLR scheduler(optimizer, 0.9f);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_lr(), 0.9f, 1e-5f);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_lr(), 0.81f, 1e-5f);
}

// ============================================================================
// CosineAnnealingLR Tests
// ============================================================================

TEST(CosineAnnealingLRTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 0.1f);

    CosineAnnealingLR scheduler(optimizer, 100, 0.001f);

    EXPECT_EQ(scheduler.T_max(), 100);
    EXPECT_FLOAT_EQ(scheduler.eta_min(), 0.001f);
}

TEST(CosineAnnealingLRTest, Cycle) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 1.0f);

    CosineAnnealingLR scheduler(optimizer, 10, 0.0f);

    // At T_max/2, LR should be approximately (base_lr + eta_min) / 2
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    // At halfway point, cosine should be 0, so LR = 0.5
    EXPECT_NEAR(scheduler.get_lr(), 0.5f, 0.1f);

    // At T_max, LR should be eta_min
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_lr(), 0.0f, 0.1f);
}

// ============================================================================
// ReduceLROnPlateau Tests
// ============================================================================

TEST(ReduceLROnPlateauTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 0.1f);

    ReduceLROnPlateau scheduler(optimizer, ReduceLROnPlateau::Mode::MIN);

    EXPECT_FLOAT_EQ(scheduler.factor(), 0.1f);
    EXPECT_EQ(scheduler.patience(), 10);
}

TEST(ReduceLROnPlateauTest, NoReduction) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 1.0f);

    ReduceLROnPlateau scheduler(optimizer, ReduceLROnPlateau::Mode::MIN, 0.1f, 2);

    // Improving metric - no reduction
    scheduler.step(1.0f);
    scheduler.step(0.9f);
    scheduler.step(0.8f);

    EXPECT_FLOAT_EQ(scheduler.get_lr(), 1.0f);
}

TEST(ReduceLROnPlateauTest, Reduction) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 1.0f);

    ReduceLROnPlateau scheduler(optimizer, ReduceLROnPlateau::Mode::MIN, 0.5f, 2);

    // Non-improving metric
    scheduler.step(1.0f);
    scheduler.step(1.0f);
    scheduler.step(1.0f);  // patience=2 exceeded

    EXPECT_NEAR(scheduler.get_lr(), 0.5f, 1e-5f);
}

// ============================================================================
// OneCycleLR Tests
// ============================================================================

TEST(OneCycleLRTest, Construction) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 0.1f);

    OneCycleLR scheduler(optimizer, 0.1f, 100);

    EXPECT_FLOAT_EQ(scheduler.max_lr(), 0.1f);
    EXPECT_EQ(scheduler.total_steps(), 100);
}

TEST(OneCycleLRTest, WarmupPhase) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 0.004f);  // Will be overwritten

    OneCycleLR scheduler(optimizer, 0.1f, 100, 0.3f);  // 30% warmup

    // During warmup, LR should increase
    float prev_lr = scheduler.get_lr();
    for (int i = 0; i < 10; ++i) {
        scheduler.step();
        EXPECT_GE(scheduler.get_lr(), prev_lr - 1e-6f);  // Non-decreasing
        prev_lr = scheduler.get_lr();
    }
}

// ============================================================================
// LinearLR Tests
// ============================================================================

TEST(LinearLRTest, Warmup) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 0.1f);

    LinearLR scheduler(optimizer, 0.1f, 1.0f, 10);  // Warmup from 0.1 to 1.0

    EXPECT_FLOAT_EQ(scheduler.start_factor(), 0.1f);
    EXPECT_FLOAT_EQ(scheduler.end_factor(), 1.0f);
    EXPECT_EQ(scheduler.total_iters(), 10);

    // After 10 steps, should reach base_lr * end_factor
    for (int i = 0; i < 10; ++i) {
        scheduler.step();
    }

    EXPECT_NEAR(scheduler.get_lr(), 0.1f, 1e-5f);  // base_lr * 1.0
}

// ============================================================================
// PolynomialLR Tests
// ============================================================================

TEST(PolynomialLRTest, LinearDecay) {
    auto model = create_test_model();
    auto params = model->parameters();
    SGD optimizer(params, 1.0f);

    PolynomialLR scheduler(optimizer, 10, 1.0f, 0.0f);  // Linear decay

    // At halfway point
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_lr(), 0.5f, 0.1f);

    // At end
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(scheduler.get_lr(), 0.0f, 0.1f);
}
