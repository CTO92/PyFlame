#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

#include "pyflame/ir/node.hpp"
#include "pyflame/ir/op_type.hpp"
#include "pyflame/autograd/grad_mode.hpp"

namespace pyflame {
class Tensor;
}

namespace pyflame::autograd {

/// Type alias for gradient function
/// Takes: grad_output (gradient of loss w.r.t. output),
///        inputs (forward pass inputs),
///        output (forward pass output)
/// Returns: gradients with respect to each input
using GradientFn = std::function<std::vector<std::shared_ptr<ir::Node>>(
    std::shared_ptr<ir::Node> grad_output,
    const std::vector<std::shared_ptr<ir::Node>>& inputs,
    std::shared_ptr<ir::Node> output,
    std::shared_ptr<ir::Graph> graph
)>;

/// Registry of gradient functions for each operation type
class GradientRegistry {
public:
    /// Get the singleton instance
    static GradientRegistry& instance();

    /// Register a gradient function for an operation type
    void register_gradient(ir::OpType op, GradientFn fn);

    /// Check if a gradient function is registered
    bool has_gradient(ir::OpType op) const;

    /// Get the gradient function for an operation type
    const GradientFn& get_gradient(ir::OpType op) const;

private:
    GradientRegistry();
    void register_all_gradients();

    std::unordered_map<ir::OpType, GradientFn> registry_;
};

/// Compute gradients through backward pass
/// This is the main entry point for autograd
class AutogradEngine {
public:
    /// Compute gradients of a scalar output with respect to all tensors
    /// that have requires_grad=true
    static void backward(
        std::shared_ptr<ir::Node> output,
        std::shared_ptr<ir::Node> grad_output,
        std::shared_ptr<ir::Graph> graph
    );

    /// Compute gradients with respect to specific inputs
    static std::vector<std::shared_ptr<ir::Node>> grad(
        const std::vector<std::shared_ptr<ir::Node>>& outputs,
        const std::vector<std::shared_ptr<ir::Node>>& inputs,
        const std::vector<std::shared_ptr<ir::Node>>& grad_outputs,
        std::shared_ptr<ir::Graph> graph,
        bool retain_graph = false,
        bool create_graph = false
    );

private:
    /// Build reverse topological order for backward pass
    static std::vector<std::shared_ptr<ir::Node>> reverse_topological_sort(
        const std::vector<std::shared_ptr<ir::Node>>& outputs
    );
};

/// Helper to create gradient nodes
namespace grad_ops {

/// Create a node that broadcasts grad to match input shape
std::shared_ptr<ir::Node> broadcast_grad_to_shape(
    std::shared_ptr<ir::Node> grad,
    const std::vector<int64_t>& target_shape,
    std::shared_ptr<ir::Graph> graph
);

/// Create a node that sums grad over broadcast dimensions
std::shared_ptr<ir::Node> sum_grad_for_broadcast(
    std::shared_ptr<ir::Node> grad,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    std::shared_ptr<ir::Graph> graph
);

/// Create a ones_like node
std::shared_ptr<ir::Node> ones_like(
    std::shared_ptr<ir::Node> node,
    std::shared_ptr<ir::Graph> graph
);

/// Create a zeros_like node
std::shared_ptr<ir::Node> zeros_like(
    std::shared_ptr<ir::Node> node,
    std::shared_ptr<ir::Graph> graph
);

}  // namespace grad_ops

}  // namespace pyflame::autograd
