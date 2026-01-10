#!/usr/bin/env python3
"""
PyFlame MLP Example

Demonstrates building a simple multi-layer perceptron (feed-forward neural network).
"""

import pyflame as pf
import numpy as np


class Linear:
    """A simple linear layer."""

    def __init__(self, in_features: int, out_features: int):
        # Initialize with small random weights
        self.weight = pf.randn([in_features, out_features]) * 0.1
        self.bias = pf.zeros([out_features])

    def __call__(self, x):
        return pf.matmul(x, self.weight) + self.bias


class MLP:
    """A simple multi-layer perceptron."""

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))

    def __call__(self, x):
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = pf.relu(layer(h))
        # No activation on last layer
        h = self.layers[-1](h)
        return h


def main():
    print("PyFlame MLP Example")
    print("=" * 40)

    # Create MLP: 784 -> 256 -> 128 -> 10 (MNIST-like)
    print("\nCreating MLP architecture...")
    model = MLP(784, [256, 128], 10)

    print("Architecture:")
    print("  Input:  784")
    print("  Hidden: 256 -> ReLU")
    print("  Hidden: 128 -> ReLU")
    print("  Output: 10")

    # Create a batch of random inputs
    batch_size = 32
    x = pf.randn([batch_size, 784])
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Evaluate
    pf.eval(output)

    # Convert to numpy for analysis
    output_np = output.numpy()

    # Show first sample's logits
    print(f"\nFirst sample logits:")
    print(f"  {output_np[0]}")

    # Apply softmax for probabilities
    probs = pf.softmax(output, dim=1)
    pf.eval(probs)
    probs_np = probs.numpy()

    print(f"\nFirst sample probabilities (softmax):")
    print(f"  {probs_np[0]}")

    # Find predicted class
    pred_class = np.argmax(probs_np[0])
    print(f"\nPredicted class: {pred_class} (probability: {probs_np[0][pred_class]:.4f})")

    # Graph analysis
    print("\nComputation graph statistics:")
    graph = pf.get_graph(output)
    print(f"  Total nodes: {graph.num_nodes()}")
    print(f"  Operations: {graph.num_ops()}")
    print(f"  Estimated memory: {graph.estimated_memory_bytes() / (1024 * 1024):.2f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
