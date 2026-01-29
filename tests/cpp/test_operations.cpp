#include <gtest/gtest.h>
#include "pyflame/core/tensor.hpp"
#include <cmath>

using namespace pyflame;

class OperationsTest : public ::testing::Test {
protected:
    static constexpr float EPSILON = 1e-5f;

    bool near_equal(float a, float b, float eps = EPSILON) {
        return std::abs(a - b) < eps;
    }
};

TEST_F(OperationsTest, Relu) {
    float values[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto x = Tensor::from_data(values, {5});
    auto y = relu(x);

    y.eval();
    const float* data = y.data<float>();

    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 1.0f);
    EXPECT_FLOAT_EQ(data[4], 2.0f);
}

TEST_F(OperationsTest, Sigmoid) {
    float values[] = {0.0f};
    auto x = Tensor::from_data(values, {1});
    auto y = sigmoid(x);

    y.eval();
    const float* data = y.data<float>();

    EXPECT_TRUE(near_equal(data[0], 0.5f));
}

TEST_F(OperationsTest, Tanh) {
    float values[] = {0.0f};
    auto x = Tensor::from_data(values, {1});
    auto y = pyflame::tanh(x);

    y.eval();
    const float* data = y.data<float>();

    EXPECT_FLOAT_EQ(data[0], 0.0f);
}

TEST_F(OperationsTest, Exp) {
    float values[] = {0.0f, 1.0f};
    auto x = Tensor::from_data(values, {2});
    auto y = pyflame::exp(x);

    y.eval();
    const float* data = y.data<float>();

    EXPECT_TRUE(near_equal(data[0], 1.0f));
    EXPECT_TRUE(near_equal(data[1], std::exp(1.0f)));
}

TEST_F(OperationsTest, Log) {
    float values[] = {1.0f, std::exp(1.0f)};
    auto x = Tensor::from_data(values, {2});
    auto y = pyflame::log(x);

    y.eval();
    const float* data = y.data<float>();

    EXPECT_TRUE(near_equal(data[0], 0.0f));
    EXPECT_TRUE(near_equal(data[1], 1.0f));
}

TEST_F(OperationsTest, Sqrt) {
    float values[] = {1.0f, 4.0f, 9.0f};
    auto x = Tensor::from_data(values, {3});
    auto y = pyflame::sqrt(x);

    y.eval();
    const float* data = y.data<float>();

    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
}

TEST_F(OperationsTest, Abs) {
    float values[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    auto x = Tensor::from_data(values, {5});
    auto y = pyflame::abs(x);

    y.eval();
    const float* data = y.data<float>();

    EXPECT_FLOAT_EQ(data[0], 3.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 1.0f);
    EXPECT_FLOAT_EQ(data[4], 3.0f);
}

TEST_F(OperationsTest, Sum) {
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor::from_data(values, {4});
    auto y = x.sum();

    y.eval();
    const float* data = y.data<float>();

    EXPECT_FLOAT_EQ(data[0], 10.0f);
}

TEST_F(OperationsTest, Mean) {
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor::from_data(values, {4});
    auto y = x.mean();

    y.eval();
    const float* data = y.data<float>();

    EXPECT_FLOAT_EQ(data[0], 2.5f);
}

TEST_F(OperationsTest, Max) {
    float values[] = {1.0f, 5.0f, 3.0f, 2.0f};
    auto x = Tensor::from_data(values, {4});
    auto y = x.max();

    y.eval();
    const float* data = y.data<float>();

    EXPECT_FLOAT_EQ(data[0], 5.0f);
}

TEST_F(OperationsTest, Min) {
    float values[] = {5.0f, 1.0f, 3.0f, 2.0f};
    auto x = Tensor::from_data(values, {4});
    auto y = x.min();

    y.eval();
    const float* data = y.data<float>();

    EXPECT_FLOAT_EQ(data[0], 1.0f);
}

TEST_F(OperationsTest, ChainedOperations) {
    auto a = Tensor::ones({10});
    auto b = Tensor::full({10}, 2.0f);

    // (a + b) * 3 - 1 = (1 + 2) * 3 - 1 = 8
    auto c = (a + b) * 3.0f - 1.0f;

    c.eval();
    const float* data = c.data<float>();

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(data[i], 8.0f);
    }
}

TEST_F(OperationsTest, MultipleReductions) {
    auto x = Tensor::arange(1, 11);  // [1, 2, ..., 10]

    auto sum = x.sum();
    auto mean = x.mean();

    sum.eval();
    mean.eval();

    EXPECT_FLOAT_EQ(sum.data<float>()[0], 55.0f);  // 1+2+...+10 = 55
    EXPECT_FLOAT_EQ(mean.data<float>()[0], 5.5f);
}

TEST_F(OperationsTest, Matmul2D) {
    // A = [[1, 2], [3, 4]] (2x2)
    // B = [[5, 6], [7, 8]] (2x2)
    // A @ B = [[19, 22], [43, 50]]
    float a_vals[] = {1, 2, 3, 4};
    float b_vals[] = {5, 6, 7, 8};

    auto a = Tensor::from_data(a_vals, {2, 2});
    auto b = Tensor::from_data(b_vals, {2, 2});
    auto c = matmul(a, b);

    c.eval();
    const float* data = c.data<float>();

    EXPECT_FLOAT_EQ(data[0], 19.0f);
    EXPECT_FLOAT_EQ(data[1], 22.0f);
    EXPECT_FLOAT_EQ(data[2], 43.0f);
    EXPECT_FLOAT_EQ(data[3], 50.0f);
}

TEST_F(OperationsTest, MatmulOperator) {
    float a_vals[] = {1, 2, 3, 4};
    float b_vals[] = {5, 6, 7, 8};

    auto a = Tensor::from_data(a_vals, {2, 2});
    auto b = Tensor::from_data(b_vals, {2, 2});
    auto c = matmul(a, b);  // Matrix multiplication

    c.eval();
    const float* data = c.data<float>();

    EXPECT_FLOAT_EQ(data[0], 19.0f);
}

TEST_F(OperationsTest, MatmulNonSquare) {
    // A = [[1, 2, 3]] (1x3)
    // B = [[1], [2], [3]] (3x1)
    // A @ B = [[14]] (dot product)
    float a_vals[] = {1, 2, 3};
    float b_vals[] = {1, 2, 3};

    auto a = Tensor::from_data(a_vals, {1, 3});
    auto b = Tensor::from_data(b_vals, {3, 1});
    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], 1);
    EXPECT_EQ(c.shape()[1], 1);

    c.eval();
    EXPECT_FLOAT_EQ(c.data<float>()[0], 14.0f);  // 1*1 + 2*2 + 3*3
}
