#include <gtest/gtest.h>
#include "pyflame/core/tensor.hpp"
#include <cmath>

using namespace pyflame;

TEST(TensorTest, Zeros) {
    auto t = Tensor::zeros({3, 4});

    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.numel(), 12);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 4);
    EXPECT_EQ(t.dtype(), DType::Float32);

    t.eval();
    const float* data = t.data<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST(TensorTest, Ones) {
    auto t = Tensor::ones({2, 3});

    EXPECT_EQ(t.numel(), 6);

    t.eval();
    const float* data = t.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST(TensorTest, Full) {
    auto t = Tensor::full({2, 2}, 3.14f);

    t.eval();
    const float* data = t.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.14f);
    }
}

TEST(TensorTest, Randn) {
    auto t = Tensor::randn({100});

    EXPECT_EQ(t.numel(), 100);

    t.eval();
    const float* data = t.data<float>();

    // Compute mean and variance
    float sum = 0, sum_sq = 0;
    for (int i = 0; i < 100; ++i) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }
    float mean = sum / 100;
    float var = sum_sq / 100 - mean * mean;

    // Should be approximately normal(0, 1)
    // With 100 samples, we expect some variance
    EXPECT_NEAR(mean, 0.0f, 0.5f);
    EXPECT_NEAR(var, 1.0f, 0.5f);
}

TEST(TensorTest, Arange) {
    auto t = Tensor::arange(0, 5);

    EXPECT_EQ(t.numel(), 5);
    EXPECT_EQ(t.ndim(), 1);

    t.eval();
    const float* data = t.data<float>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

TEST(TensorTest, FromData) {
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_data(values, {2, 2});

    EXPECT_EQ(t.numel(), 4);

    t.eval();
    const float* data = t.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST(TensorTest, Addition) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::full({2, 3}, 2.0f);
    auto c = a + b;

    EXPECT_EQ(c.shape(), a.shape());

    c.eval();
    const float* data = c.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f);
    }
}

TEST(TensorTest, Multiplication) {
    auto a = Tensor::full({2, 2}, 3.0f);
    auto b = Tensor::full({2, 2}, 4.0f);
    auto c = a * b;

    c.eval();
    const float* data = c.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], 12.0f);
    }
}

TEST(TensorTest, ScalarOperations) {
    auto a = Tensor::ones({3});
    auto b = a + 2.0f;
    auto c = a * 3.0f;

    b.eval();
    c.eval();

    const float* b_data = b.data<float>();
    const float* c_data = c.data<float>();

    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(b_data[i], 3.0f);
        EXPECT_FLOAT_EQ(c_data[i], 3.0f);
    }
}

TEST(TensorTest, Negation) {
    auto a = Tensor::full({2, 2}, 5.0f);
    auto b = -a;

    b.eval();
    const float* data = b.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], -5.0f);
    }
}

TEST(TensorTest, Reshape) {
    auto a = Tensor::arange(0, 6);
    auto b = a.reshape({2, 3});

    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);
    EXPECT_EQ(b.numel(), 6);
}

TEST(TensorTest, Transpose) {
    float values[] = {1, 2, 3, 4, 5, 6};
    auto a = Tensor::from_data(values, {2, 3});
    auto b = a.t();

    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape()[1], 2);
}

TEST(TensorTest, LazyEvaluation) {
    auto a = Tensor::ones({100});
    auto b = Tensor::ones({100});
    auto c = a + b;  // Not evaluated yet

    EXPECT_FALSE(c.is_evaluated());

    c.eval();

    EXPECT_TRUE(c.is_evaluated());
}

TEST(TensorTest, ToString) {
    auto t = Tensor::zeros({3, 4});
    std::string s = t.to_string();

    EXPECT_TRUE(s.find("Tensor") != std::string::npos);
    EXPECT_TRUE(s.find("3") != std::string::npos);
    EXPECT_TRUE(s.find("4") != std::string::npos);
}
