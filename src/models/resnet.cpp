#include "pyflame/models/resnet.hpp"
#include "pyflame/ops/activations.hpp"

namespace pyflame::models {

// ============================================================================
// BasicBlock Implementation
// ============================================================================

BasicBlock::BasicBlock(
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride,
    nn::Module* downsample
) : conv1_(in_channels, out_channels, 3, stride, 1, 1, false),
    bn1_(out_channels),
    conv2_(out_channels, out_channels, 3, 1, 1, 1, false),
    bn2_(out_channels),
    stride_(stride)
{
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);
    register_module("conv2", conv2_);
    register_module("bn2", bn2_);

    if (downsample) {
        downsample_.reset(downsample);
        register_module("downsample", *downsample_);
    }
}

Tensor BasicBlock::forward(const Tensor& x) {
    Tensor identity = x;

    // First conv block
    Tensor out = conv1_.forward(x);
    out = bn1_.forward(out);
    out = ops::relu(out);

    // Second conv block
    out = conv2_.forward(out);
    out = bn2_.forward(out);

    // Skip connection
    if (downsample_) {
        identity = downsample_->forward(x);
    }

    out = out + identity;
    out = ops::relu(out);

    return out;
}

// ============================================================================
// Bottleneck Implementation
// ============================================================================

Bottleneck::Bottleneck(
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride,
    nn::Module* downsample,
    int64_t groups,
    int64_t base_width
) : width_(out_channels * base_width / 64 * groups),
    conv1_(in_channels, width_, 1, 1, 0, 1, false),
    bn1_(width_),
    conv2_(width_, width_, 3, stride, 1, groups, false),
    bn2_(width_),
    conv3_(width_, out_channels * EXPANSION, 1, 1, 0, 1, false),
    bn3_(out_channels * EXPANSION),
    stride_(stride)
{
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);
    register_module("conv2", conv2_);
    register_module("bn2", bn2_);
    register_module("conv3", conv3_);
    register_module("bn3", bn3_);

    if (downsample) {
        downsample_.reset(downsample);
        register_module("downsample", *downsample_);
    }
}

Tensor Bottleneck::forward(const Tensor& x) {
    Tensor identity = x;

    // 1x1 reduce
    Tensor out = conv1_.forward(x);
    out = bn1_.forward(out);
    out = ops::relu(out);

    // 3x3 conv
    out = conv2_.forward(out);
    out = bn2_.forward(out);
    out = ops::relu(out);

    // 1x1 expand
    out = conv3_.forward(out);
    out = bn3_.forward(out);

    // Skip connection
    if (downsample_) {
        identity = downsample_->forward(x);
    }

    out = out + identity;
    out = ops::relu(out);

    return out;
}

// ============================================================================
// ResNetConfig Implementations
// ============================================================================

ResNetConfig ResNetConfig::ResNet18(int64_t num_classes) {
    return ResNetConfig{
        .name = "ResNet18",
        .layers = {2, 2, 2, 2},
        .use_bottleneck = false,
        .num_classes = num_classes
    };
}

ResNetConfig ResNetConfig::ResNet34(int64_t num_classes) {
    return ResNetConfig{
        .name = "ResNet34",
        .layers = {3, 4, 6, 3},
        .use_bottleneck = false,
        .num_classes = num_classes
    };
}

ResNetConfig ResNetConfig::ResNet50(int64_t num_classes) {
    return ResNetConfig{
        .name = "ResNet50",
        .layers = {3, 4, 6, 3},
        .use_bottleneck = true,
        .num_classes = num_classes
    };
}

ResNetConfig ResNetConfig::ResNet101(int64_t num_classes) {
    return ResNetConfig{
        .name = "ResNet101",
        .layers = {3, 4, 23, 3},
        .use_bottleneck = true,
        .num_classes = num_classes
    };
}

ResNetConfig ResNetConfig::ResNet152(int64_t num_classes) {
    return ResNetConfig{
        .name = "ResNet152",
        .layers = {3, 8, 36, 3},
        .use_bottleneck = true,
        .num_classes = num_classes
    };
}

ResNetConfig ResNetConfig::ResNeXt50_32x4d(int64_t num_classes) {
    return ResNetConfig{
        .name = "ResNeXt50_32x4d",
        .layers = {3, 4, 6, 3},
        .use_bottleneck = true,
        .num_classes = num_classes,
        .groups = 32,
        .width_per_group = 4
    };
}

ResNetConfig ResNetConfig::ResNeXt101_32x8d(int64_t num_classes) {
    return ResNetConfig{
        .name = "ResNeXt101_32x8d",
        .layers = {3, 4, 23, 3},
        .use_bottleneck = true,
        .num_classes = num_classes,
        .groups = 32,
        .width_per_group = 8
    };
}

ResNetConfig ResNetConfig::WideResNet50_2(int64_t num_classes) {
    return ResNetConfig{
        .name = "WideResNet50_2",
        .layers = {3, 4, 6, 3},
        .use_bottleneck = true,
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 128
    };
}

ResNetConfig ResNetConfig::WideResNet101_2(int64_t num_classes) {
    return ResNetConfig{
        .name = "WideResNet101_2",
        .layers = {3, 4, 23, 3},
        .use_bottleneck = true,
        .num_classes = num_classes,
        .groups = 1,
        .width_per_group = 128
    };
}

// ============================================================================
// ResNet Implementation
// ============================================================================

ResNet::ResNet(const ResNetConfig& config)
    : config_(config),
      conv1_(3, 64, 7, 2, 3, 1, false),
      bn1_(64),
      maxpool_(3, 2, 1),
      avgpool_({1, 1}),
      fc_(512 * (config.use_bottleneck ? Bottleneck::EXPANSION : BasicBlock::EXPANSION),
          config.num_classes)
{
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);
    register_module("maxpool", maxpool_);

    // Build residual stages
    if (config_.use_bottleneck) {
        layer1_ = make_layer<Bottleneck>(64, config_.layers[0], 1);
        layer2_ = make_layer<Bottleneck>(128, config_.layers[1], 2);
        layer3_ = make_layer<Bottleneck>(256, config_.layers[2], 2);
        layer4_ = make_layer<Bottleneck>(512, config_.layers[3], 2);
        num_features_ = 512 * Bottleneck::EXPANSION;
    } else {
        layer1_ = make_layer<BasicBlock>(64, config_.layers[0], 1);
        layer2_ = make_layer<BasicBlock>(128, config_.layers[1], 2);
        layer3_ = make_layer<BasicBlock>(256, config_.layers[2], 2);
        layer4_ = make_layer<BasicBlock>(512, config_.layers[3], 2);
        num_features_ = 512 * BasicBlock::EXPANSION;
    }

    register_module("layer1", layer1_);
    register_module("layer2", layer2_);
    register_module("layer3", layer3_);
    register_module("layer4", layer4_);
    register_module("avgpool", avgpool_);
    register_module("fc", fc_);

    // Initialize weights
    for (auto& [name, param] : named_parameters()) {
        if (name.find("weight") != std::string::npos) {
            // Kaiming initialization for conv weights
            // param = nn::init::kaiming_normal(param, 0, "fan_out", "relu");
        } else if (name.find("bias") != std::string::npos) {
            // Zero initialization for biases
            // param = Tensor::zeros_like(param);
        }
    }

    // Zero-initialize the last BN in each residual branch
    if (config_.zero_init_residual) {
        for (auto& [name, param] : named_parameters()) {
            if (name.find("bn2.weight") != std::string::npos ||
                name.find("bn3.weight") != std::string::npos) {
                // param = Tensor::zeros_like(param);
            }
        }
    }
}

template<typename BlockType>
nn::Sequential ResNet::make_layer(
    int64_t planes,
    int blocks,
    int64_t stride
) {
    nn::Sequential layers;

    // Determine if we need a downsample path
    std::unique_ptr<nn::Sequential> downsample_ptr;
    int64_t expansion = BlockType::EXPANSION;

    if (stride != 1 || in_planes_ != planes * expansion) {
        // Create downsample sequential: conv1x1 + bn
        // Use unique_ptr for exception safety and automatic cleanup
        downsample_ptr = std::make_unique<nn::Sequential>();
        downsample_ptr->add(std::make_shared<nn::Conv2d>(
            in_planes_, planes * expansion, 1, stride, 0, 1, false));
        downsample_ptr->add(std::make_shared<nn::BatchNorm2d>(planes * expansion));
    }

    // First block may have stride and downsample
    // Release ownership to the block (which will manage it via unique_ptr)
    nn::Module* downsample = downsample_ptr.release();

    if constexpr (std::is_same_v<BlockType, Bottleneck>) {
        layers.add(std::make_shared<Bottleneck>(
            in_planes_, planes, stride, downsample,
            config_.groups, config_.width_per_group));
    } else {
        layers.add(std::make_shared<BasicBlock>(
            in_planes_, planes, stride, downsample));
    }

    in_planes_ = planes * expansion;

    // Remaining blocks
    for (int i = 1; i < blocks; ++i) {
        if constexpr (std::is_same_v<BlockType, Bottleneck>) {
            layers.add(std::make_shared<Bottleneck>(
                in_planes_, planes, 1, nullptr,
                config_.groups, config_.width_per_group));
        } else {
            layers.add(std::make_shared<BasicBlock>(
                in_planes_, planes, 1, nullptr));
        }
    }

    return layers;
}

// Explicit template instantiations
template nn::Sequential ResNet::make_layer<BasicBlock>(int64_t, int, int64_t);
template nn::Sequential ResNet::make_layer<Bottleneck>(int64_t, int, int64_t);

Tensor ResNet::forward(const Tensor& x) {
    // Get features
    Tensor out = forward_features(x);

    // Classifier
    out = avgpool_.forward(out);
    out = out.flatten(1);  // Flatten all dims except batch
    out = fc_.forward(out);

    return out;
}

Tensor ResNet::forward_features(const Tensor& x) {
    // Initial convolution
    Tensor out = conv1_.forward(x);
    out = bn1_.forward(out);
    out = ops::relu(out);
    out = maxpool_.forward(out);

    // Residual stages
    out = layer1_.forward(out);
    out = layer2_.forward(out);
    out = layer3_.forward(out);
    out = layer4_.forward(out);

    return out;
}

void ResNet::reset_classifier(int64_t num_classes) {
    config_.num_classes = num_classes;
    fc_ = nn::Linear(num_features_, num_classes);
    register_module("fc", fc_);
}

}  // namespace pyflame::models
