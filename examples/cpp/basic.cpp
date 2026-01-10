/**
 * PyFlame Basic Example
 *
 * Demonstrates basic tensor operations and the lazy evaluation model.
 */

#include <iostream>
#include "pyflame/pyflame.hpp"

using namespace pyflame;

int main() {
    std::cout << "PyFlame v" << VERSION << " Basic Example\n";
    std::cout << "========================================\n\n";

    // Create tensors
    std::cout << "Creating tensors...\n";
    auto a = Tensor::randn({3, 4});
    auto b = Tensor::randn({3, 4});

    std::cout << "a: " << a << "\n";
    std::cout << "b: " << b << "\n";

    // Arithmetic operations (lazy - not computed yet)
    std::cout << "\nBuilding computation graph...\n";
    auto c = a + b;
    auto d = c * 2.0f;
    auto e = relu(d);
    auto f = e.sum();

    std::cout << "Operations recorded. Graph not yet executed.\n";
    std::cout << "f.is_evaluated(): " << std::boolalpha << f.is_evaluated() << "\n";

    // Force evaluation
    std::cout << "\nEvaluating...\n";
    f.eval();

    std::cout << "f.is_evaluated(): " << f.is_evaluated() << "\n";
    std::cout << "Result: " << f.data<float>()[0] << "\n";

    // Access the computation graph
    std::cout << "\nComputation graph:\n";
    auto graph = f.graph();
    std::cout << graph->to_string() << "\n";

    std::cout << "\nGraph statistics:\n";
    std::cout << "  Nodes: " << graph->num_nodes() << "\n";
    std::cout << "  Operations: " << graph->num_ops() << "\n";
    std::cout << "  Estimated memory: " << graph->estimated_memory_bytes() << " bytes\n";

    return 0;
}
