/**
 * PyFlame MLP Example
 *
 * Demonstrates building a simple multi-layer perceptron (feed-forward neural network).
 */

#include <iostream>
#include <vector>
#include "pyflame/pyflame.hpp"

using namespace pyflame;

/// A simple linear layer
class Linear {
public:
    Linear(int in_features, int out_features)
        : weight(Tensor::randn({in_features, out_features}) * 0.1f),
          bias(Tensor::zeros({out_features})) {}

    Tensor forward(const Tensor& x) const {
        return matmul(x, weight) + bias;
    }

    Tensor weight;
    Tensor bias;
};

/// A simple MLP
class MLP {
public:
    MLP(int input_size, const std::vector<int>& hidden_sizes, int output_size) {
        std::vector<int> sizes;
        sizes.push_back(input_size);
        for (int h : hidden_sizes) sizes.push_back(h);
        sizes.push_back(output_size);

        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            layers.emplace_back(sizes[i], sizes[i + 1]);
        }
    }

    Tensor forward(const Tensor& x) const {
        Tensor h = x;
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            h = relu(layers[i].forward(h));
        }
        // No activation on last layer
        h = layers.back().forward(h);
        return h;
    }

    std::vector<Linear> layers;
};

int main() {
    std::cout << "PyFlame MLP Example\n";
    std::cout << "===================\n\n";

    // Create MLP: 784 -> 256 -> 128 -> 10
    std::cout << "Creating MLP architecture...\n";
    MLP model(784, {256, 128}, 10);

    std::cout << "Architecture:\n";
    std::cout << "  Input:  784\n";
    std::cout << "  Hidden: 256 -> ReLU\n";
    std::cout << "  Hidden: 128 -> ReLU\n";
    std::cout << "  Output: 10\n\n";

    // Create a batch of random inputs
    int batch_size = 32;
    auto x = Tensor::randn({batch_size, 784});

    std::cout << "Input shape: [" << x.shape()[0] << ", " << x.shape()[1] << "]\n";

    // Forward pass
    std::cout << "\nRunning forward pass...\n";
    auto output = model.forward(x);

    std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]\n";

    // Evaluate
    output.eval();

    // Show first sample's logits
    std::cout << "\nFirst sample logits:\n  [";
    const float* data = output.data<float>();
    for (int i = 0; i < 10; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << data[i];
    }
    std::cout << "]\n";

    // Apply softmax for probabilities
    auto probs = softmax(output, 1);
    probs.eval();

    std::cout << "\nFirst sample probabilities (softmax):\n  [";
    const float* prob_data = probs.data<float>();
    for (int i = 0; i < 10; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << prob_data[i];
    }
    std::cout << "]\n";

    // Find predicted class
    int pred_class = 0;
    float max_prob = prob_data[0];
    for (int i = 1; i < 10; ++i) {
        if (prob_data[i] > max_prob) {
            max_prob = prob_data[i];
            pred_class = i;
        }
    }
    std::cout << "\nPredicted class: " << pred_class << " (probability: " << max_prob << ")\n";

    // Graph analysis
    std::cout << "\nComputation graph statistics:\n";
    auto graph = output.graph();
    std::cout << "  Total nodes: " << graph->num_nodes() << "\n";
    std::cout << "  Operations: " << graph->num_ops() << "\n";
    std::cout << "  Estimated memory: " << graph->estimated_memory_bytes() / (1024 * 1024.0)
              << " MB\n";

    std::cout << "\nDone!\n";
    return 0;
}
