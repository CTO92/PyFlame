// Phase 2 Tests: Loss Functions

#include <gtest/gtest.h>
#include "pyflame/core/tensor.hpp"
#include "pyflame/nn/loss.hpp"
#include <cmath>

using namespace pyflame;
using namespace pyflame::nn;

// ============================================================================
// MSELoss Tests
// ============================================================================

TEST(MSELossTest, PerfectPrediction) {
    MSELoss loss;

    auto pred = Tensor::full({4}, 1.0f);
    auto target = Tensor::full({4}, 1.0f);

    auto l = loss(pred, target);
    l.eval();

    EXPECT_NEAR(l.data<float>()[0], 0.0f, 1e-6f);
}

TEST(MSELossTest, KnownValues) {
    MSELoss loss(Reduction::MEAN);

    // pred = [0, 0], target = [1, 2]
    // MSE = ((0-1)^2 + (0-2)^2) / 2 = (1 + 4) / 2 = 2.5
    float pred_data[] = {0.0f, 0.0f};
    float target_data[] = {1.0f, 2.0f};

    auto pred = Tensor::from_data(pred_data, {2});
    auto target = Tensor::from_data(target_data, {2});

    auto l = loss(pred, target);
    l.eval();

    EXPECT_NEAR(l.data<float>()[0], 2.5f, 1e-5f);
}

TEST(MSELossTest, ReductionSum) {
    MSELoss loss(Reduction::SUM);

    float pred_data[] = {0.0f, 0.0f};
    float target_data[] = {1.0f, 2.0f};

    auto pred = Tensor::from_data(pred_data, {2});
    auto target = Tensor::from_data(target_data, {2});

    auto l = loss(pred, target);
    l.eval();

    // Sum = 1 + 4 = 5
    EXPECT_NEAR(l.data<float>()[0], 5.0f, 1e-5f);
}

TEST(MSELossTest, ReductionNone) {
    MSELoss loss(Reduction::NONE);

    float pred_data[] = {0.0f, 0.0f};
    float target_data[] = {1.0f, 2.0f};

    auto pred = Tensor::from_data(pred_data, {2});
    auto target = Tensor::from_data(target_data, {2});

    auto l = loss(pred, target);
    l.eval();

    EXPECT_EQ(l.numel(), 2);
    EXPECT_NEAR(l.data<float>()[0], 1.0f, 1e-5f);  // (0-1)^2 = 1
    EXPECT_NEAR(l.data<float>()[1], 4.0f, 1e-5f);  // (0-2)^2 = 4
}

// ============================================================================
// L1Loss Tests
// ============================================================================

TEST(L1LossTest, PerfectPrediction) {
    L1Loss loss;

    auto pred = Tensor::full({4}, 2.0f);
    auto target = Tensor::full({4}, 2.0f);

    auto l = loss(pred, target);
    l.eval();

    EXPECT_NEAR(l.data<float>()[0], 0.0f, 1e-6f);
}

TEST(L1LossTest, KnownValues) {
    L1Loss loss(Reduction::MEAN);

    // L1 = (|0-1| + |0-2|) / 2 = (1 + 2) / 2 = 1.5
    float pred_data[] = {0.0f, 0.0f};
    float target_data[] = {1.0f, 2.0f};

    auto pred = Tensor::from_data(pred_data, {2});
    auto target = Tensor::from_data(target_data, {2});

    auto l = loss(pred, target);
    l.eval();

    EXPECT_NEAR(l.data<float>()[0], 1.5f, 1e-5f);
}

// ============================================================================
// SmoothL1Loss Tests
// ============================================================================

TEST(SmoothL1LossTest, SmallErrors) {
    // For |x| < beta, L = 0.5 * x^2 / beta
    SmoothL1Loss loss(Reduction::MEAN, 1.0f);

    float pred_data[] = {0.0f};
    float target_data[] = {0.5f};  // |error| = 0.5 < 1.0

    auto pred = Tensor::from_data(pred_data, {1});
    auto target = Tensor::from_data(target_data, {1});

    auto l = loss(pred, target);
    l.eval();

    // L = 0.5 * 0.5^2 / 1.0 = 0.125
    EXPECT_NEAR(l.data<float>()[0], 0.125f, 1e-5f);
}

// ============================================================================
// BCELoss Tests
// ============================================================================

TEST(BCELossTest, KnownValues) {
    BCELoss loss(Reduction::MEAN);

    // BCE = -[y * log(p) + (1-y) * log(1-p)]
    // For p=0.5, y=1: BCE = -[1 * log(0.5)] = -log(0.5) = log(2) ≈ 0.693
    float pred_data[] = {0.5f};
    float target_data[] = {1.0f};

    auto pred = Tensor::from_data(pred_data, {1});
    auto target = Tensor::from_data(target_data, {1});

    auto l = loss(pred, target);
    l.eval();

    EXPECT_NEAR(l.data<float>()[0], std::log(2.0f), 1e-4f);
}

TEST(BCELossTest, PerfectPrediction) {
    BCELoss loss(Reduction::MEAN);

    // For p≈1, y=1: BCE ≈ 0
    float pred_data[] = {0.9999f};
    float target_data[] = {1.0f};

    auto pred = Tensor::from_data(pred_data, {1});
    auto target = Tensor::from_data(target_data, {1});

    auto l = loss(pred, target);
    l.eval();

    // Should be close to 0
    EXPECT_LT(l.data<float>()[0], 0.01f);
}

// ============================================================================
// CrossEntropyLoss Tests
// ============================================================================

TEST(CrossEntropyLossTest, Construction) {
    CrossEntropyLoss loss;
    std::string s = loss.to_string();

    EXPECT_TRUE(s.find("CrossEntropyLoss") != std::string::npos);
}

TEST(CrossEntropyLossTest, LabelSmoothing) {
    CrossEntropyLoss loss_no_smooth(Reduction::MEAN, -100, 0.0f);
    CrossEntropyLoss loss_smooth(Reduction::MEAN, -100, 0.1f);

    // Label smoothing should produce different results
    EXPECT_NE(loss_no_smooth.to_string(), loss_smooth.to_string());
}

// ============================================================================
// KLDivLoss Tests
// ============================================================================

TEST(KLDivLossTest, Construction) {
    KLDivLoss loss;
    std::string s = loss.to_string();

    EXPECT_TRUE(s.find("KLDivLoss") != std::string::npos);
}

// ============================================================================
// CosineEmbeddingLoss Tests
// ============================================================================

TEST(CosineEmbeddingLossTest, IdenticalVectors) {
    CosineEmbeddingLoss loss;

    auto x1 = Tensor::full({3}, 1.0f);
    auto x2 = Tensor::full({3}, 1.0f);
    auto target = Tensor::full({1}, 1.0f);  // Similar pair

    auto l = loss.forward(x1, x2, target);
    l.eval();

    // Identical vectors have cosine similarity = 1
    // For y=1: L = 1 - cos = 1 - 1 = 0
    EXPECT_NEAR(l.data<float>()[0], 0.0f, 1e-5f);
}

// ============================================================================
// TripletMarginLoss Tests
// ============================================================================

TEST(TripletMarginLossTest, Construction) {
    TripletMarginLoss loss(Reduction::MEAN, 1.0f, 2.0f);
    std::string s = loss.to_string();

    EXPECT_TRUE(s.find("TripletMarginLoss") != std::string::npos);
    EXPECT_TRUE(s.find("margin=1") != std::string::npos);
}

// ============================================================================
// Functional Interface Tests
// ============================================================================

TEST(FunctionalLossTest, MSE) {
    auto pred = Tensor::zeros({4});
    auto target = Tensor::ones({4});

    auto l = functional::mse_loss(pred, target, Reduction::MEAN);
    l.eval();

    // MSE of (0-1)^2 = 1 for all elements
    EXPECT_NEAR(l.data<float>()[0], 1.0f, 1e-5f);
}

TEST(FunctionalLossTest, L1) {
    auto pred = Tensor::zeros({4});
    auto target = Tensor::full({4}, 2.0f);

    auto l = functional::l1_loss(pred, target, Reduction::MEAN);
    l.eval();

    // L1 = |0-2| = 2 for all elements
    EXPECT_NEAR(l.data<float>()[0], 2.0f, 1e-5f);
}
