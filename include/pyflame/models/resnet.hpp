#pragma once

#include "pyflame/nn/module.hpp"
#include "pyflame/nn/layers.hpp"
#include "pyflame/core/tensor.hpp"

#include <vector>
#include <memory>
#include <string>

namespace pyflame::models {

// ============================================================================
// ResNet Building Blocks
// ============================================================================

/// Basic residual block for ResNet-18 and ResNet-34
/// Two 3x3 conv layers with skip connection
class BasicBlock : public nn::Module {
public:
    static constexpr int EXPANSION = 1;

    BasicBlock(
        int64_t in_channels,
        int64_t out_channels,
        int64_t stride = 1,
        nn::Module* downsample = nullptr
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "BasicBlock"; }

private:
    nn::Conv2d conv1_;
    nn::BatchNorm2d bn1_;
    nn::Conv2d conv2_;
    nn::BatchNorm2d bn2_;
    std::unique_ptr<nn::Module> downsample_;
    int64_t stride_;
};

/// Bottleneck residual block for ResNet-50, ResNet-101, ResNet-152
/// 1x1 -> 3x3 -> 1x1 conv with skip connection
class Bottleneck : public nn::Module {
public:
    static constexpr int EXPANSION = 4;

    Bottleneck(
        int64_t in_channels,
        int64_t out_channels,
        int64_t stride = 1,
        nn::Module* downsample = nullptr,
        int64_t groups = 1,
        int64_t base_width = 64
    );

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "Bottleneck"; }

private:
    int64_t width_;
    nn::Conv2d conv1_;
    nn::BatchNorm2d bn1_;
    nn::Conv2d conv2_;
    nn::BatchNorm2d bn2_;
    nn::Conv2d conv3_;
    nn::BatchNorm2d bn3_;
    std::unique_ptr<nn::Module> downsample_;
    int64_t stride_;
};

// ============================================================================
// ResNet Configuration
// ============================================================================

/// ResNet variant configurations
struct ResNetConfig {
    std::string name;
    std::vector<int> layers;  // Number of blocks in each stage
    bool use_bottleneck;      // Use Bottleneck vs BasicBlock
    int64_t num_classes = 1000;
    int64_t groups = 1;
    int64_t width_per_group = 64;
    bool zero_init_residual = false;

    // Predefined configurations
    static ResNetConfig ResNet18(int64_t num_classes = 1000);
    static ResNetConfig ResNet34(int64_t num_classes = 1000);
    static ResNetConfig ResNet50(int64_t num_classes = 1000);
    static ResNetConfig ResNet101(int64_t num_classes = 1000);
    static ResNetConfig ResNet152(int64_t num_classes = 1000);
    static ResNetConfig ResNeXt50_32x4d(int64_t num_classes = 1000);
    static ResNetConfig ResNeXt101_32x8d(int64_t num_classes = 1000);
    static ResNetConfig WideResNet50_2(int64_t num_classes = 1000);
    static ResNetConfig WideResNet101_2(int64_t num_classes = 1000);
};

// ============================================================================
// ResNet Model
// ============================================================================

/// ResNet model supporting all standard variants
/// Reference: "Deep Residual Learning for Image Recognition"
/// https://arxiv.org/abs/1512.03385
class ResNet : public nn::Module {
public:
    /// Construct ResNet from configuration
    explicit ResNet(const ResNetConfig& config);

    /// Forward pass
    Tensor forward(const Tensor& x) override;

    /// Get feature maps before classifier (for transfer learning)
    Tensor forward_features(const Tensor& x);

    /// Get the number of output features from the backbone
    int64_t num_features() const { return num_features_; }

    /// Get number of classes
    int64_t num_classes() const { return config_.num_classes; }

    std::string name() const override { return config_.name; }

    /// Replace the classifier head for transfer learning
    void reset_classifier(int64_t num_classes);

private:
    ResNetConfig config_;
    int64_t in_planes_ = 64;
    int64_t num_features_;

    // Initial layers
    nn::Conv2d conv1_;
    nn::BatchNorm2d bn1_;
    nn::MaxPool2d maxpool_;

    // Residual stages
    nn::Sequential layer1_;
    nn::Sequential layer2_;
    nn::Sequential layer3_;
    nn::Sequential layer4_;

    // Classifier
    nn::AdaptiveAvgPool2d avgpool_;
    nn::Linear fc_;

    // Helper to create a residual stage
    template<typename BlockType>
    nn::Sequential make_layer(
        int64_t planes,
        int blocks,
        int64_t stride = 1
    );
};

// ============================================================================
// Convenience Factory Functions
// ============================================================================

/// Create ResNet-18 model
inline std::unique_ptr<ResNet> resnet18(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::ResNet18(num_classes));
}

/// Create ResNet-34 model
inline std::unique_ptr<ResNet> resnet34(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::ResNet34(num_classes));
}

/// Create ResNet-50 model
inline std::unique_ptr<ResNet> resnet50(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::ResNet50(num_classes));
}

/// Create ResNet-101 model
inline std::unique_ptr<ResNet> resnet101(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::ResNet101(num_classes));
}

/// Create ResNet-152 model
inline std::unique_ptr<ResNet> resnet152(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::ResNet152(num_classes));
}

/// Create ResNeXt-50 32x4d model
inline std::unique_ptr<ResNet> resnext50_32x4d(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::ResNeXt50_32x4d(num_classes));
}

/// Create ResNeXt-101 32x8d model
inline std::unique_ptr<ResNet> resnext101_32x8d(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::ResNeXt101_32x8d(num_classes));
}

/// Create Wide ResNet-50-2 model
inline std::unique_ptr<ResNet> wide_resnet50_2(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::WideResNet50_2(num_classes));
}

/// Create Wide ResNet-101-2 model
inline std::unique_ptr<ResNet> wide_resnet101_2(int64_t num_classes = 1000) {
    return std::make_unique<ResNet>(ResNetConfig::WideResNet101_2(num_classes));
}

}  // namespace pyflame::models
