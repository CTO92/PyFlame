#include <gtest/gtest.h>
#include "pyflame/core/dtype.hpp"

using namespace pyflame;

TEST(DTypeTest, SizeBytes) {
    EXPECT_EQ(dtype_size(DType::Float32), 4);
    EXPECT_EQ(dtype_size(DType::Float16), 2);
    EXPECT_EQ(dtype_size(DType::BFloat16), 2);
    EXPECT_EQ(dtype_size(DType::Int32), 4);
    EXPECT_EQ(dtype_size(DType::Int16), 2);
    EXPECT_EQ(dtype_size(DType::Int8), 1);
    EXPECT_EQ(dtype_size(DType::Bool), 1);
}

TEST(DTypeTest, Names) {
    EXPECT_EQ(dtype_name(DType::Float32), "float32");
    EXPECT_EQ(dtype_name(DType::Float16), "float16");
    EXPECT_EQ(dtype_name(DType::BFloat16), "bfloat16");
    EXPECT_EQ(dtype_name(DType::Int32), "int32");
}

TEST(DTypeTest, IsFloating) {
    EXPECT_TRUE(dtype_is_floating(DType::Float32));
    EXPECT_TRUE(dtype_is_floating(DType::Float16));
    EXPECT_TRUE(dtype_is_floating(DType::BFloat16));
    EXPECT_FALSE(dtype_is_floating(DType::Int32));
    EXPECT_FALSE(dtype_is_floating(DType::Bool));
}

TEST(DTypeTest, IsInteger) {
    EXPECT_TRUE(dtype_is_integer(DType::Int32));
    EXPECT_TRUE(dtype_is_integer(DType::Int16));
    EXPECT_TRUE(dtype_is_integer(DType::Int8));
    EXPECT_FALSE(dtype_is_integer(DType::Float32));
    EXPECT_FALSE(dtype_is_integer(DType::Bool));
}

TEST(DTypeTest, CSLConversion) {
    EXPECT_EQ(dtype_to_csl(DType::Float32), "f32");
    EXPECT_EQ(dtype_to_csl(DType::Float16), "f16");
    EXPECT_EQ(dtype_to_csl(DType::Int32), "i32");
}
