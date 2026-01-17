#include "pyflame/autograd/autograd.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/ir/shape_inference.hpp"

#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

namespace pyflame::autograd {

// ============================================================================
// GradientRegistry Implementation
// ============================================================================

GradientRegistry& GradientRegistry::instance() {
    static GradientRegistry instance;
    return instance;
}

GradientRegistry::GradientRegistry() {
    register_all_gradients();
}

void GradientRegistry::register_gradient(ir::OpType op, GradientFn fn) {
    registry_[op] = std::move(fn);
}

bool GradientRegistry::has_gradient(ir::OpType op) const {
    return registry_.find(op) != registry_.end();
}

const GradientFn& GradientRegistry::get_gradient(ir::OpType op) const {
    auto it = registry_.find(op);
    if (it == registry_.end()) {
        throw std::runtime_error("No gradient registered for op: " + ir::op_type_name(op));
    }
    return it->second;
}

void GradientRegistry::register_all_gradients() {
    using namespace ir;

    // ========================================================================
    // Elementwise Binary Operations
    // ========================================================================

    // Add: d(a+b)/da = 1, d(a+b)/db = 1
    register_gradient(OpType::ADD, [](auto grad_out, auto inputs, auto, auto graph) {
        auto grad_a = grad_ops::sum_grad_for_broadcast(grad_out, inputs[0]->shape(), grad_out->shape(), graph);
        auto grad_b = grad_ops::sum_grad_for_broadcast(grad_out, inputs[1]->shape(), grad_out->shape(), graph);
        return std::vector<std::shared_ptr<Node>>{grad_a, grad_b};
    });

    // Sub: d(a-b)/da = 1, d(a-b)/db = -1
    register_gradient(OpType::SUB, [](auto grad_out, auto inputs, auto, auto graph) {
        auto grad_a = grad_ops::sum_grad_for_broadcast(grad_out, inputs[0]->shape(), grad_out->shape(), graph);

        // Create -grad_out
        auto neg_grad = graph->create_op(OpType::NEG, {grad_out},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        auto grad_b = grad_ops::sum_grad_for_broadcast(neg_grad, inputs[1]->shape(), grad_out->shape(), graph);

        return std::vector<std::shared_ptr<Node>>{grad_a, grad_b};
    });

    // Mul: d(a*b)/da = b, d(a*b)/db = a
    register_gradient(OpType::MUL, [](auto grad_out, auto inputs, auto, auto graph) {
        // grad_a = grad_out * b
        auto mul_a = graph->create_op(OpType::MUL, {grad_out, inputs[1]},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        auto grad_a = grad_ops::sum_grad_for_broadcast(mul_a, inputs[0]->shape(), grad_out->shape(), graph);

        // grad_b = grad_out * a
        auto mul_b = graph->create_op(OpType::MUL, {grad_out, inputs[0]},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        auto grad_b = grad_ops::sum_grad_for_broadcast(mul_b, inputs[1]->shape(), grad_out->shape(), graph);

        return std::vector<std::shared_ptr<Node>>{grad_a, grad_b};
    });

    // Div: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    register_gradient(OpType::DIV, [](auto grad_out, auto inputs, auto, auto graph) {
        auto a = inputs[0];
        auto b = inputs[1];

        // grad_a = grad_out / b
        auto div_result = graph->create_op(OpType::DIV, {grad_out, b},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        auto grad_a = grad_ops::sum_grad_for_broadcast(div_result, a->shape(), grad_out->shape(), graph);

        // grad_b = -grad_out * a / (b * b)
        auto neg_grad = graph->create_op(OpType::NEG, {grad_out},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        auto mul_a = graph->create_op(OpType::MUL, {neg_grad, a},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        auto b_sq = graph->create_op(OpType::MUL, {b, b},
            ir::TensorSpec(b->shape(), b->dtype(), b->layout()));
        auto div_b = graph->create_op(OpType::DIV, {mul_a, b_sq},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        auto grad_b = grad_ops::sum_grad_for_broadcast(div_b, b->shape(), grad_out->shape(), graph);

        return std::vector<std::shared_ptr<Node>>{grad_a, grad_b};
    });

    // ========================================================================
    // Elementwise Unary Operations
    // ========================================================================

    // Neg: d(-x)/dx = -1
    register_gradient(OpType::NEG, [](auto grad_out, auto inputs, auto, auto graph) {
        auto grad = graph->create_op(OpType::NEG, {grad_out},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Abs: d|x|/dx = sign(x)
    register_gradient(OpType::ABS, [](auto grad_out, auto inputs, auto, auto graph) {
        auto x = inputs[0];
        auto zero = grad_ops::zeros_like(x, graph);

        // sign(x) = (x > 0) - (x < 0)
        auto gt = graph->create_op(OpType::GT, {x, zero},
            ir::TensorSpec(x->shape(), x->dtype(), x->layout()));
        auto lt = graph->create_op(OpType::LT, {x, zero},
            ir::TensorSpec(x->shape(), x->dtype(), x->layout()));
        auto sign = graph->create_op(OpType::SUB, {gt, lt},
            ir::TensorSpec(x->shape(), x->dtype(), x->layout()));

        auto grad = graph->create_op(OpType::MUL, {grad_out, sign},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));

        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Sqrt: d(sqrt(x))/dx = 0.5 / sqrt(x)
    register_gradient(OpType::SQRT, [](auto grad_out, auto inputs, auto output, auto graph) {
        // grad = grad_out * 0.5 / output
        auto half = graph->create_constant(
            ir::TensorSpec({}, grad_out->dtype(), grad_out->layout()));

        auto two_output = graph->create_op(OpType::MUL, {output, output},
            ir::TensorSpec(output->shape(), output->dtype(), output->layout()));
        // Actually: grad = grad_out / (2 * sqrt(x)) = grad_out / (2 * output)

        auto two = graph->create_constant(ir::TensorSpec({}, grad_out->dtype(), grad_out->layout()));
        // Set constant data for 2.0
        float two_val = 2.0f;
        std::vector<uint8_t> two_data(sizeof(float));
        std::memcpy(two_data.data(), &two_val, sizeof(float));
        two->set_constant_data(std::move(two_data));

        auto two_times_output = graph->create_op(OpType::MUL, {two, output},
            ir::TensorSpec(output->shape(), output->dtype(), output->layout()));
        auto grad = graph->create_op(OpType::DIV, {grad_out, two_times_output},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));

        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Exp: d(exp(x))/dx = exp(x)
    register_gradient(OpType::EXP, [](auto grad_out, auto inputs, auto output, auto graph) {
        // grad = grad_out * output (since output = exp(x))
        auto grad = graph->create_op(OpType::MUL, {grad_out, output},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Log: d(log(x))/dx = 1/x
    register_gradient(OpType::LOG, [](auto grad_out, auto inputs, auto, auto graph) {
        auto x = inputs[0];
        auto grad = graph->create_op(OpType::DIV, {grad_out, x},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Sin: d(sin(x))/dx = cos(x)
    register_gradient(OpType::SIN, [](auto grad_out, auto inputs, auto, auto graph) {
        auto x = inputs[0];
        auto cos_x = graph->create_op(OpType::COS, {x},
            ir::TensorSpec(x->shape(), x->dtype(), x->layout()));
        auto grad = graph->create_op(OpType::MUL, {grad_out, cos_x},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Cos: d(cos(x))/dx = -sin(x)
    register_gradient(OpType::COS, [](auto grad_out, auto inputs, auto, auto graph) {
        auto x = inputs[0];
        auto sin_x = graph->create_op(OpType::SIN, {x},
            ir::TensorSpec(x->shape(), x->dtype(), x->layout()));
        auto neg_sin = graph->create_op(OpType::NEG, {sin_x},
            ir::TensorSpec(x->shape(), x->dtype(), x->layout()));
        auto grad = graph->create_op(OpType::MUL, {grad_out, neg_sin},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Tanh: d(tanh(x))/dx = 1 - tanh(x)^2
    register_gradient(OpType::TANH, [](auto grad_out, auto inputs, auto output, auto graph) {
        // grad = grad_out * (1 - output^2)
        auto one = grad_ops::ones_like(output, graph);
        auto out_sq = graph->create_op(OpType::MUL, {output, output},
            ir::TensorSpec(output->shape(), output->dtype(), output->layout()));
        auto one_minus = graph->create_op(OpType::SUB, {one, out_sq},
            ir::TensorSpec(output->shape(), output->dtype(), output->layout()));
        auto grad = graph->create_op(OpType::MUL, {grad_out, one_minus},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // ========================================================================
    // Activation Functions
    // ========================================================================

    // ReLU: d(relu(x))/dx = 1 if x > 0, else 0
    register_gradient(OpType::RELU, [](auto grad_out, auto inputs, auto, auto graph) {
        auto x = inputs[0];
        auto zero = grad_ops::zeros_like(x, graph);
        auto mask = graph->create_op(OpType::GT, {x, zero},
            ir::TensorSpec(x->shape(), x->dtype(), x->layout()));
        auto grad = graph->create_op(OpType::MUL, {grad_out, mask},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
    register_gradient(OpType::SIGMOID, [](auto grad_out, auto inputs, auto output, auto graph) {
        auto one = grad_ops::ones_like(output, graph);
        auto one_minus_y = graph->create_op(OpType::SUB, {one, output},
            ir::TensorSpec(output->shape(), output->dtype(), output->layout()));
        auto y_times_one_minus_y = graph->create_op(OpType::MUL, {output, one_minus_y},
            ir::TensorSpec(output->shape(), output->dtype(), output->layout()));
        auto grad = graph->create_op(OpType::MUL, {grad_out, y_times_one_minus_y},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Softmax: complex gradient, see Phase 2 doc
    register_gradient(OpType::SOFTMAX, [](auto grad_out, auto inputs, auto output, auto graph) {
        // grad = y * (grad_out - sum(grad_out * y, dim))
        int dim = output->get_attr<int>("dim", -1);

        // Normalize negative dimension
        int ndim = static_cast<int>(grad_out->shape().size());
        if (dim < 0) dim += ndim;

        auto gy = graph->create_op(OpType::MUL, {grad_out, output},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));

        // Sum along the softmax dimension (use normalized dim)
        auto gy_sum = graph->create_op(OpType::SUM, {gy},
            ir::infer_reduction_shape(gy->shape(), dim, true));
        gy_sum->set_attr("dim", dim);
        gy_sum->set_attr("keepdim", true);

        auto grad_minus_sum = graph->create_op(OpType::SUB, {grad_out, gy_sum},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));

        auto grad = graph->create_op(OpType::MUL, {output, grad_minus_sum},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));

        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // ========================================================================
    // Reduction Operations
    // ========================================================================

    // Sum: broadcast gradient back to input shape
    register_gradient(OpType::SUM, [](auto grad_out, auto inputs, auto output, auto graph) {
        auto input_shape = inputs[0]->shape();

        // If grad_out is scalar or reduced, broadcast back
        auto grad = grad_ops::broadcast_grad_to_shape(grad_out, input_shape, graph);

        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // Mean: broadcast gradient / n back to input shape
    register_gradient(OpType::MEAN, [](auto grad_out, auto inputs, auto output, auto graph) {
        auto input_shape = inputs[0]->shape();

        // Compute number of elements in reduction
        int dim = output->get_attr<int>("dim", -1);
        int ndim = static_cast<int>(input_shape.size());

        // Normalize negative dimension
        if (dim < 0) dim += ndim;

        int64_t n;
        if (dim < 0 || dim >= ndim) {
            // Full reduction
            n = inputs[0]->numel();
        } else {
            n = input_shape[dim];
        }

        // Check for division by zero
        if (n == 0) {
            // Return zero gradient for empty tensor
            auto zero_grad = graph->create_constant(
                ir::TensorSpec(input_shape, grad_out->dtype(), grad_out->layout()));
            return std::vector<std::shared_ptr<Node>>{zero_grad};
        }

        // Create 1/n constant
        auto inv_n = graph->create_constant(ir::TensorSpec({}, grad_out->dtype(), grad_out->layout()));
        float inv_n_val = 1.0f / static_cast<float>(n);
        std::vector<uint8_t> inv_n_data(sizeof(float));
        std::memcpy(inv_n_data.data(), &inv_n_val, sizeof(float));
        inv_n->set_constant_data(std::move(inv_n_data));

        // Scale gradient
        auto scaled_grad = graph->create_op(OpType::MUL, {grad_out, inv_n},
            ir::TensorSpec(grad_out->shape(), grad_out->dtype(), grad_out->layout()));

        auto grad = grad_ops::broadcast_grad_to_shape(scaled_grad, input_shape, graph);

        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // ========================================================================
    // Matrix Operations
    // ========================================================================

    // MatMul: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    register_gradient(OpType::MATMUL, [](auto grad_out, auto inputs, auto, auto graph) {
        auto A = inputs[0];
        auto B = inputs[1];

        // Transpose B: B^T
        auto B_T = graph->create_op(OpType::TRANSPOSE, {B},
            ir::TensorSpec({B->shape()[1], B->shape()[0]}, B->dtype(), B->layout()));
        B_T->set_attr("dim0", 0);
        B_T->set_attr("dim1", 1);

        // Transpose A: A^T
        auto A_T = graph->create_op(OpType::TRANSPOSE, {A},
            ir::TensorSpec({A->shape()[1], A->shape()[0]}, A->dtype(), A->layout()));
        A_T->set_attr("dim0", 0);
        A_T->set_attr("dim1", 1);

        // grad_A = grad_out @ B^T
        auto grad_A = graph->create_op(OpType::MATMUL, {grad_out, B_T},
            ir::TensorSpec(A->shape(), A->dtype(), A->layout()));

        // grad_B = A^T @ grad_out
        auto grad_B = graph->create_op(OpType::MATMUL, {A_T, grad_out},
            ir::TensorSpec(B->shape(), B->dtype(), B->layout()));

        return std::vector<std::shared_ptr<Node>>{grad_A, grad_B};
    });

    // Transpose: just transpose the gradient back
    register_gradient(OpType::TRANSPOSE, [](auto grad_out, auto inputs, auto output, auto graph) {
        int dim0 = output->get_attr<int>("dim0");
        int dim1 = output->get_attr<int>("dim1");

        auto grad = graph->create_op(OpType::TRANSPOSE, {grad_out},
            ir::TensorSpec(inputs[0]->shape(), grad_out->dtype(), grad_out->layout()));
        grad->set_attr("dim0", dim0);
        grad->set_attr("dim1", dim1);

        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // ========================================================================
    // Shape Operations
    // ========================================================================

    // Reshape: reshape gradient back to input shape
    register_gradient(OpType::RESHAPE, [](auto grad_out, auto inputs, auto, auto graph) {
        auto input_shape = inputs[0]->shape();
        auto grad = graph->create_op(OpType::RESHAPE, {grad_out},
            ir::TensorSpec(input_shape, grad_out->dtype(), grad_out->layout()));
        grad->set_attr("new_shape", input_shape);
        return std::vector<std::shared_ptr<Node>>{grad};
    });

    // View: same as reshape
    register_gradient(OpType::VIEW, [](auto grad_out, auto inputs, auto, auto graph) {
        auto input_shape = inputs[0]->shape();
        auto grad = graph->create_op(OpType::VIEW, {grad_out},
            ir::TensorSpec(input_shape, grad_out->dtype(), grad_out->layout()));
        return std::vector<std::shared_ptr<Node>>{grad};
    });
}

// ============================================================================
// AutogradEngine Implementation
// ============================================================================

std::vector<std::shared_ptr<ir::Node>> AutogradEngine::reverse_topological_sort(
    const std::vector<std::shared_ptr<ir::Node>>& outputs
) {
    std::vector<std::shared_ptr<ir::Node>> result;
    std::unordered_set<ir::NodeId> visited;

    std::function<void(const std::shared_ptr<ir::Node>&)> visit =
        [&](const std::shared_ptr<ir::Node>& node) {
            if (visited.count(node->id())) return;
            visited.insert(node->id());

            for (const auto& input : node->inputs()) {
                visit(input);
            }
            result.push_back(node);
        };

    for (const auto& out : outputs) {
        visit(out);
    }

    // Reverse for backward pass order
    std::reverse(result.begin(), result.end());
    return result;
}

void AutogradEngine::backward(
    std::shared_ptr<ir::Node> output,
    std::shared_ptr<ir::Node> grad_output,
    std::shared_ptr<ir::Graph> graph
) {
    if (!grad_output) {
        // Default gradient of 1 for scalar outputs
        grad_output = grad_ops::ones_like(output, graph);
    }

    // Map from node to accumulated gradient
    std::unordered_map<ir::NodeId, std::shared_ptr<ir::Node>> grad_map;
    grad_map[output->id()] = grad_output;

    // Get reverse topological order
    auto order = reverse_topological_sort({output});

    // Process nodes in reverse order
    for (const auto& node : order) {
        if (!node->is_operation()) continue;

        auto it = grad_map.find(node->id());
        if (it == grad_map.end()) continue;  // No gradient flows to this node

        auto grad_out = it->second;

        // Get gradient function
        auto& registry = GradientRegistry::instance();
        if (!registry.has_gradient(node->op_type())) {
            continue;  // No gradient for this op (e.g., comparison ops)
        }

        // Compute input gradients
        auto grad_fn = registry.get_gradient(node->op_type());
        auto input_grads = grad_fn(grad_out, node->inputs(), node, graph);

        // Accumulate gradients to inputs
        for (size_t i = 0; i < input_grads.size() && i < node->inputs().size(); ++i) {
            if (!input_grads[i]) continue;

            auto input_id = node->inputs()[i]->id();
            auto git = grad_map.find(input_id);
            if (git == grad_map.end()) {
                grad_map[input_id] = input_grads[i];
            } else {
                // Accumulate gradients
                auto sum = graph->create_op(ir::OpType::ADD, {git->second, input_grads[i]},
                    ir::TensorSpec(input_grads[i]->shape(), input_grads[i]->dtype(), input_grads[i]->layout()));
                grad_map[input_id] = sum;
            }
        }
    }

    // Store gradients in parameter/input nodes
    for (auto& [node_id, grad] : grad_map) {
        auto node = graph->node(node_id);
        if (node && (node->is_parameter() || node->is_input())) {
            node->set_attr("grad", grad);
        }
    }
}

std::vector<std::shared_ptr<ir::Node>> AutogradEngine::grad(
    const std::vector<std::shared_ptr<ir::Node>>& outputs,
    const std::vector<std::shared_ptr<ir::Node>>& inputs,
    const std::vector<std::shared_ptr<ir::Node>>& grad_outputs,
    std::shared_ptr<ir::Graph> graph,
    bool retain_graph,
    bool create_graph
) {
    // Initialize gradient map
    std::unordered_map<ir::NodeId, std::shared_ptr<ir::Node>> grad_map;

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto grad_out = (i < grad_outputs.size() && grad_outputs[i])
            ? grad_outputs[i]
            : grad_ops::ones_like(outputs[i], graph);
        grad_map[outputs[i]->id()] = grad_out;
    }

    // Get reverse topological order
    auto order = reverse_topological_sort(outputs);

    // Process nodes
    for (const auto& node : order) {
        if (!node->is_operation()) continue;

        auto it = grad_map.find(node->id());
        if (it == grad_map.end()) continue;

        auto grad_out = it->second;

        auto& registry = GradientRegistry::instance();
        if (!registry.has_gradient(node->op_type())) continue;

        auto grad_fn = registry.get_gradient(node->op_type());
        auto input_grads = grad_fn(grad_out, node->inputs(), node, graph);

        for (size_t i = 0; i < input_grads.size() && i < node->inputs().size(); ++i) {
            if (!input_grads[i]) continue;

            auto input_id = node->inputs()[i]->id();
            auto git = grad_map.find(input_id);
            if (git == grad_map.end()) {
                grad_map[input_id] = input_grads[i];
            } else {
                auto sum = graph->create_op(ir::OpType::ADD, {git->second, input_grads[i]},
                    ir::TensorSpec(input_grads[i]->shape(), input_grads[i]->dtype(), input_grads[i]->layout()));
                grad_map[input_id] = sum;
            }
        }
    }

    // Collect gradients for requested inputs
    std::vector<std::shared_ptr<ir::Node>> result;
    for (const auto& input : inputs) {
        auto it = grad_map.find(input->id());
        result.push_back(it != grad_map.end() ? it->second : nullptr);
    }

    return result;
}

// ============================================================================
// grad_ops Implementation
// ============================================================================

namespace grad_ops {

std::shared_ptr<ir::Node> broadcast_grad_to_shape(
    std::shared_ptr<ir::Node> grad,
    const std::vector<int64_t>& target_shape,
    std::shared_ptr<ir::Graph> graph
) {
    if (grad->shape() == target_shape) {
        return grad;
    }

    // Create broadcast op
    auto result = graph->create_op(ir::OpType::BROADCAST, {grad},
        ir::TensorSpec(target_shape, grad->dtype(), grad->layout()));
    result->set_attr("target_shape", target_shape);

    return result;
}

std::shared_ptr<ir::Node> sum_grad_for_broadcast(
    std::shared_ptr<ir::Node> grad,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    std::shared_ptr<ir::Graph> graph
) {
    if (input_shape == output_shape) {
        return grad;
    }

    // Find dimensions that were broadcast
    // For simplicity, if shapes differ, sum over the extra dims
    auto result = grad;

    // Handle case where input was scalar (empty shape or single element)
    if (input_shape.empty() || (input_shape.size() == 1 && input_shape[0] == 1)) {
        result = graph->create_op(ir::OpType::SUM, {grad},
            ir::TensorSpec(input_shape.empty() ? std::vector<int64_t>{} : input_shape,
                          grad->dtype(), grad->layout()));
        return result;
    }

    // Sum over broadcast dimensions
    int ndim_diff = static_cast<int>(output_shape.size()) - static_cast<int>(input_shape.size());

    // Sum over leading dimensions if they were added
    for (int i = 0; i < ndim_diff; ++i) {
        result = graph->create_op(ir::OpType::SUM, {result},
            ir::infer_reduction_shape(result->shape(), 0, false));
        result->set_attr("dim", 0);
        result->set_attr("keepdim", false);
    }

    // Sum over dimensions that were broadcast (size 1 -> size n)
    auto res_shape = result->shape();
    for (int i = static_cast<int>(input_shape.size()) - 1; i >= 0; --i) {
        if (input_shape[i] == 1 && res_shape[i] != 1) {
            result = graph->create_op(ir::OpType::SUM, {result},
                ir::infer_reduction_shape(result->shape(), i, true));
            result->set_attr("dim", i);
            result->set_attr("keepdim", true);
        }
    }

    return result;
}

std::shared_ptr<ir::Node> ones_like(
    std::shared_ptr<ir::Node> node,
    std::shared_ptr<ir::Graph> graph
) {
    auto spec = ir::TensorSpec(node->shape(), node->dtype(), node->layout());
    auto result = graph->create_constant(spec);

    // Fill with ones
    size_t bytes = spec.size_bytes();
    int64_t numel = spec.numel();
    std::vector<uint8_t> data(bytes);

    if (node->dtype() == DType::Float32) {
        float* ptr = reinterpret_cast<float*>(data.data());
        std::fill(ptr, ptr + numel, 1.0f);
    }

    result->set_constant_data(std::move(data));
    return result;
}

std::shared_ptr<ir::Node> zeros_like(
    std::shared_ptr<ir::Node> node,
    std::shared_ptr<ir::Graph> graph
) {
    auto spec = ir::TensorSpec(node->shape(), node->dtype(), node->layout());
    auto result = graph->create_constant(spec);

    // Data is already zeroed by default
    size_t bytes = spec.size_bytes();
    std::vector<uint8_t> data(bytes, 0);
    result->set_constant_data(std::move(data));

    return result;
}

}  // namespace grad_ops

}  // namespace pyflame::autograd
