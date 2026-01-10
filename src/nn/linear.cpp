#include "pyflame/nn/linear.hpp"

#include <cmath>
#include <sstream>

namespace pyflame::nn {

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features)
    , out_features_(out_features)
    , use_bias_(bias)
{
    set_name("Linear");

    // Kaiming/He initialization for ReLU networks
    // std = sqrt(2 / fan_in)
    float std = std::sqrt(2.0f / static_cast<float>(in_features));

    // Weight: [out_features, in_features]
    weight_ = Tensor::randn({out_features, in_features}) * std;
    register_parameter("weight", weight_);

    // Bias: [out_features]
    if (use_bias_) {
        bias_ = Tensor::zeros({out_features});
        register_parameter("bias", bias_);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // input: [..., in_features]
    // weight: [out_features, in_features]
    // output: [..., out_features]

    // y = x @ weight^T
    auto weight_t = weight_.transpose(0, 1);  // [in_features, out_features]
    auto output = matmul(input, weight_t);

    if (use_bias_) {
        output = output + bias_;
    }

    return output;
}

std::string Linear::to_string() const {
    std::ostringstream oss;
    oss << "Linear(in_features=" << in_features_
        << ", out_features=" << out_features_
        << ", bias=" << (use_bias_ ? "true" : "false")
        << ")";
    return oss.str();
}

}  // namespace pyflame::nn
