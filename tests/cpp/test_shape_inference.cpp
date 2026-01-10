#include <gtest/gtest.h>
#include "pyflame/ir/shape_inference.hpp"

using namespace pyflame;
using namespace pyflame::ir;

TEST(ShapeInferenceTest, BroadcastSameShape) {
    std::vector<int64_t> a = {3, 4, 5};
    std::vector<int64_t> b = {3, 4, 5};

    auto result = broadcast_shapes(a, b);

    EXPECT_EQ(result, a);
}

TEST(ShapeInferenceTest, BroadcastScalar) {
    std::vector<int64_t> a = {3, 4};
    std::vector<int64_t> b = {};  // scalar

    auto result = broadcast_shapes(a, b);

    EXPECT_EQ(result, a);
}

TEST(ShapeInferenceTest, BroadcastExpandDim) {
    std::vector<int64_t> a = {3, 4};
    std::vector<int64_t> b = {4};  // broadcast along first dim

    auto result = broadcast_shapes(a, b);

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 4);
}

TEST(ShapeInferenceTest, BroadcastBothExpand) {
    std::vector<int64_t> a = {3, 1};
    std::vector<int64_t> b = {1, 4};

    auto result = broadcast_shapes(a, b);

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 4);
}

TEST(ShapeInferenceTest, BroadcastDifferentRank) {
    std::vector<int64_t> a = {2, 3, 4};
    std::vector<int64_t> b = {4};

    auto result = broadcast_shapes(a, b);

    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
}

TEST(ShapeInferenceTest, BroadcastIncompatible) {
    std::vector<int64_t> a = {3, 4};
    std::vector<int64_t> b = {5};  // 5 != 4 and 5 != 1

    EXPECT_THROW(broadcast_shapes(a, b), std::runtime_error);
}

TEST(ShapeInferenceTest, BinaryOpSpec) {
    TensorSpec a({3, 4}, DType::Float32);
    TensorSpec b({3, 4}, DType::Float32);

    auto result = infer_binary_op_spec(a, b, OpType::ADD);

    EXPECT_EQ(result.shape, a.shape);
    EXPECT_EQ(result.dtype, DType::Float32);
}

TEST(ShapeInferenceTest, BinaryOpBroadcast) {
    TensorSpec a({3, 4}, DType::Float32);
    TensorSpec b({4}, DType::Float32);

    auto result = infer_binary_op_spec(a, b, OpType::ADD);

    EXPECT_EQ(result.shape[0], 3);
    EXPECT_EQ(result.shape[1], 4);
}

TEST(ShapeInferenceTest, ReductionFullSpec) {
    TensorSpec input({3, 4, 5}, DType::Float32);

    auto result = infer_reduction_spec(input, std::nullopt, false);

    EXPECT_TRUE(result.shape.empty());  // Scalar
}

TEST(ShapeInferenceTest, ReductionFullKeepdim) {
    TensorSpec input({3, 4, 5}, DType::Float32);

    auto result = infer_reduction_spec(input, std::nullopt, true);

    EXPECT_EQ(result.shape.size(), 3);
    EXPECT_EQ(result.shape[0], 1);
    EXPECT_EQ(result.shape[1], 1);
    EXPECT_EQ(result.shape[2], 1);
}

TEST(ShapeInferenceTest, ReductionDimSpec) {
    TensorSpec input({3, 4, 5}, DType::Float32);

    auto result = infer_reduction_spec(input, 1, false);

    EXPECT_EQ(result.shape.size(), 2);
    EXPECT_EQ(result.shape[0], 3);
    EXPECT_EQ(result.shape[1], 5);
}

TEST(ShapeInferenceTest, ReductionDimKeepdim) {
    TensorSpec input({3, 4, 5}, DType::Float32);

    auto result = infer_reduction_spec(input, 1, true);

    EXPECT_EQ(result.shape.size(), 3);
    EXPECT_EQ(result.shape[0], 3);
    EXPECT_EQ(result.shape[1], 1);
    EXPECT_EQ(result.shape[2], 5);
}

TEST(ShapeInferenceTest, MatmulSpec2D) {
    TensorSpec a({3, 4}, DType::Float32);
    TensorSpec b({4, 5}, DType::Float32);

    auto result = infer_matmul_spec(a, b);

    EXPECT_EQ(result.shape.size(), 2);
    EXPECT_EQ(result.shape[0], 3);
    EXPECT_EQ(result.shape[1], 5);
}

TEST(ShapeInferenceTest, MatmulSpecIncompatible) {
    TensorSpec a({3, 4}, DType::Float32);
    TensorSpec b({5, 6}, DType::Float32);  // 4 != 5

    EXPECT_THROW(infer_matmul_spec(a, b), std::runtime_error);
}

TEST(ShapeInferenceTest, TransposeSpec) {
    TensorSpec input({3, 4, 5}, DType::Float32);

    auto result = infer_transpose_spec(input, 0, 2);

    EXPECT_EQ(result.shape[0], 5);
    EXPECT_EQ(result.shape[1], 4);
    EXPECT_EQ(result.shape[2], 3);
}

TEST(ShapeInferenceTest, ReshapeSpec) {
    TensorSpec input({12}, DType::Float32);

    auto result = infer_reshape_spec(input, {3, 4});

    EXPECT_EQ(result.shape[0], 3);
    EXPECT_EQ(result.shape[1], 4);
}

TEST(ShapeInferenceTest, ReshapeSpecInfer) {
    TensorSpec input({12}, DType::Float32);

    auto result = infer_reshape_spec(input, {3, -1});

    EXPECT_EQ(result.shape[0], 3);
    EXPECT_EQ(result.shape[1], 4);
}

TEST(ShapeInferenceTest, ConcatSpec) {
    std::vector<TensorSpec> inputs = {
        TensorSpec({2, 4}, DType::Float32),
        TensorSpec({3, 4}, DType::Float32),
        TensorSpec({5, 4}, DType::Float32),
    };

    auto result = infer_concat_spec(inputs, 0);

    EXPECT_EQ(result.shape[0], 10);  // 2 + 3 + 5
    EXPECT_EQ(result.shape[1], 4);
}
