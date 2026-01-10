// Phase 2 Tests: Neural Network Layers

#include <gtest/gtest.h>
#include "pyflame/core/tensor.hpp"
#include "pyflame/nn/module.hpp"
#include "pyflame/nn/linear.hpp"
#include "pyflame/nn/conv.hpp"
#include "pyflame/nn/normalization.hpp"
#include "pyflame/nn/pooling.hpp"
#include "pyflame/nn/dropout.hpp"
#include "pyflame/nn/attention.hpp"
#include <cmath>

using namespace pyflame;
using namespace pyflame::nn;

// ============================================================================
// Linear Layer Tests
// ============================================================================

TEST(LinearTest, Construction) {
    auto linear = std::make_shared<Linear>(10, 5);

    EXPECT_EQ(linear->in_features(), 10);
    EXPECT_EQ(linear->out_features(), 5);
    EXPECT_EQ(linear->name(), "Linear");
}

TEST(LinearTest, Forward) {
    auto linear = std::make_shared<Linear>(4, 2, false);  // No bias

    auto input = Tensor::randn({3, 4});  // Batch of 3, features 4
    auto output = linear->forward(input);

    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], 2);
}

TEST(LinearTest, Parameters) {
    auto linear = std::make_shared<Linear>(4, 2, true);  // With bias

    auto params = linear->parameters();

    // Should have weight and bias
    EXPECT_EQ(params.size(), 2);
}

TEST(LinearTest, ToString) {
    auto linear = std::make_shared<Linear>(10, 5);
    std::string s = linear->to_string();

    EXPECT_TRUE(s.find("Linear") != std::string::npos);
    EXPECT_TRUE(s.find("10") != std::string::npos);
    EXPECT_TRUE(s.find("5") != std::string::npos);
}

// ============================================================================
// Conv2d Tests
// ============================================================================

TEST(Conv2dTest, Construction) {
    auto conv = std::make_shared<Conv2d>(3, 16, std::array<int64_t, 2>{3, 3});

    EXPECT_EQ(conv->in_channels(), 3);
    EXPECT_EQ(conv->out_channels(), 16);
    EXPECT_EQ(conv->kernel_size()[0], 3);
    EXPECT_EQ(conv->kernel_size()[1], 3);
}

TEST(Conv2dTest, Forward) {
    auto conv = std::make_shared<Conv2d>(
        1, 8,
        std::array<int64_t, 2>{3, 3},
        std::array<int64_t, 2>{1, 1},  // stride
        std::array<int64_t, 2>{1, 1}   // padding
    );

    auto input = Tensor::randn({2, 1, 8, 8});  // Batch 2, 1 channel, 8x8
    auto output = conv->forward(input);

    // With padding=1 and stride=1, output should be same spatial size
    EXPECT_EQ(output.shape()[0], 2);   // Batch
    EXPECT_EQ(output.shape()[1], 8);   // Out channels
    EXPECT_EQ(output.shape()[2], 8);   // Height (preserved with padding)
    EXPECT_EQ(output.shape()[3], 8);   // Width (preserved with padding)
}

// ============================================================================
// BatchNorm2d Tests
// ============================================================================

TEST(BatchNorm2dTest, Construction) {
    auto bn = std::make_shared<BatchNorm2d>(64);

    EXPECT_EQ(bn->num_features(), 64);
}

TEST(BatchNorm2dTest, Forward) {
    auto bn = std::make_shared<BatchNorm2d>(16);
    bn->eval();  // Use running stats in eval mode

    auto input = Tensor::randn({4, 16, 8, 8});
    auto output = bn->forward(input);

    EXPECT_EQ(output.shape(), input.shape());
}

TEST(BatchNorm2dTest, TrainEvalModes) {
    auto bn = std::make_shared<BatchNorm2d>(8);

    EXPECT_TRUE(bn->is_training());  // Default is training mode

    bn->eval();
    EXPECT_FALSE(bn->is_training());

    bn->train();
    EXPECT_TRUE(bn->is_training());
}

// ============================================================================
// LayerNorm Tests
// ============================================================================

TEST(LayerNormTest, Construction) {
    auto ln = std::make_shared<LayerNorm>(std::vector<int64_t>{128});

    EXPECT_EQ(ln->name(), "LayerNorm");
}

TEST(LayerNormTest, Forward) {
    auto ln = std::make_shared<LayerNorm>(std::vector<int64_t>{64});

    auto input = Tensor::randn({4, 10, 64});  // [batch, seq, features]
    auto output = ln->forward(input);

    EXPECT_EQ(output.shape(), input.shape());
}

// ============================================================================
// Pooling Tests
// ============================================================================

TEST(MaxPool2dTest, Forward) {
    auto pool = std::make_shared<MaxPool2d>(
        std::array<int64_t, 2>{2, 2},  // kernel_size
        std::array<int64_t, 2>{2, 2}   // stride
    );

    auto input = Tensor::randn({1, 3, 8, 8});
    auto output = pool->forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);  // 8/2 = 4
    EXPECT_EQ(output.shape()[3], 4);
}

TEST(AvgPool2dTest, Forward) {
    auto pool = std::make_shared<AvgPool2d>(
        std::array<int64_t, 2>{2, 2},
        std::array<int64_t, 2>{2, 2}
    );

    auto input = Tensor::randn({2, 16, 16, 16});
    auto output = pool->forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST(AdaptiveAvgPool2dTest, Forward) {
    auto pool = std::make_shared<AdaptiveAvgPool2d>(std::array<int64_t, 2>{1, 1});

    auto input = Tensor::randn({4, 256, 7, 7});
    auto output = pool->forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 256);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST(GlobalAvgPool2dTest, Forward) {
    auto pool = std::make_shared<GlobalAvgPool2d>();

    auto input = Tensor::randn({2, 64, 14, 14});
    auto output = pool->forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

// ============================================================================
// Dropout Tests
// ============================================================================

TEST(DropoutTest, Construction) {
    auto dropout = std::make_shared<Dropout>(0.5f);

    EXPECT_EQ(dropout->name(), "Dropout");
}

TEST(DropoutTest, EvalModePassthrough) {
    auto dropout = std::make_shared<Dropout>(0.5f);
    dropout->eval();

    auto input = Tensor::ones({10, 10});
    auto output = dropout->forward(input);

    output.eval();
    input.eval();

    // In eval mode, dropout should pass through unchanged
    const float* in_data = input.data<float>();
    const float* out_data = output.data<float>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(in_data[i], out_data[i]);
    }
}

// ============================================================================
// Sequential Tests
// ============================================================================

TEST(SequentialTest, Forward) {
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(10, 20));
    model->add(std::make_shared<Linear>(20, 5));

    EXPECT_EQ(model->size(), 2);

    auto input = Tensor::randn({4, 10});
    auto output = model->forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 5);
}

TEST(SequentialTest, InitializerList) {
    std::vector<std::shared_ptr<Module>> layers = {
        std::make_shared<Linear>(10, 20),
        std::make_shared<Linear>(20, 10)
    };

    auto model = std::make_shared<Sequential>(layers);
    EXPECT_EQ(model->size(), 2);
}

TEST(SequentialTest, Parameters) {
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(10, 5, true));   // 2 params
    model->add(std::make_shared<Linear>(5, 2, true));    // 2 params

    auto params = model->parameters();
    EXPECT_EQ(params.size(), 4);
}

// ============================================================================
// MultiheadAttention Tests
// ============================================================================

TEST(MultiheadAttentionTest, Construction) {
    auto mha = std::make_shared<MultiheadAttention>(64, 8);

    EXPECT_EQ(mha->embed_dim(), 64);
    EXPECT_EQ(mha->num_heads(), 8);
}

TEST(MultiheadAttentionTest, SelfAttention) {
    auto mha = std::make_shared<MultiheadAttention>(32, 4);

    auto x = Tensor::randn({10, 2, 32});  // [seq, batch, embed]
    auto output = mha->forward(x);

    EXPECT_EQ(output.shape()[0], 10);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 32);
}

TEST(MultiheadAttentionTest, CrossAttention) {
    auto mha = std::make_shared<MultiheadAttention>(64, 8);

    auto query = Tensor::randn({5, 2, 64});   // [seq_q, batch, embed]
    auto key = Tensor::randn({10, 2, 64});    // [seq_k, batch, embed]
    auto value = Tensor::randn({10, 2, 64});

    auto [output, attn_weights] = mha->forward(query, key, value, Tensor(), true);

    EXPECT_EQ(output.shape()[0], 5);   // seq_q
    EXPECT_EQ(output.shape()[1], 2);   // batch
    EXPECT_EQ(output.shape()[2], 64);  // embed

    // Attention weights: [batch, seq_q, seq_k]
    EXPECT_EQ(attn_weights.shape()[0], 2);
    EXPECT_EQ(attn_weights.shape()[1], 5);
    EXPECT_EQ(attn_weights.shape()[2], 10);
}

// ============================================================================
// State Dict Tests
// ============================================================================

TEST(ModuleTest, StateDict) {
    auto linear = std::make_shared<Linear>(4, 2, true);

    auto state = linear->state_dict();

    EXPECT_TRUE(state.count("weight") > 0);
    EXPECT_TRUE(state.count("bias") > 0);
}

TEST(ModuleTest, LoadStateDict) {
    auto linear1 = std::make_shared<Linear>(4, 2, false);
    auto linear2 = std::make_shared<Linear>(4, 2, false);

    // Copy state from linear1 to linear2
    linear2->load_state_dict(linear1->state_dict());

    // States should now be identical
    auto input = Tensor::randn({1, 4});
    auto out1 = linear1->forward(input);
    auto out2 = linear2->forward(input);

    out1.eval();
    out2.eval();

    const float* data1 = out1.data<float>();
    const float* data2 = out2.data<float>();

    for (int i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]);
    }
}
