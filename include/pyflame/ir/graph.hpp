#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <sstream>

#include "pyflame/ir/node.hpp"

namespace pyflame::ir {

/// Computation graph
class Graph : public std::enable_shared_from_this<Graph> {
public:
    Graph() = default;
    ~Graph() = default;

    // Disable copy
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    // Enable move
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    /// Create a constant node
    std::shared_ptr<Node> create_constant(
        const TensorSpec& spec,
        const void* data = nullptr,
        const std::string& name = ""
    ) {
        auto node = std::make_shared<Node>(next_id_++, NodeType::CONSTANT, name);
        node->set_output_spec(spec);
        node->set_op_type(OpType::CONSTANT);

        if (data) {
            size_t bytes = spec.size_bytes();
            std::vector<uint8_t> data_copy(bytes);
            std::memcpy(data_copy.data(), data, bytes);
            node->set_constant_data(std::move(data_copy));
        }

        all_nodes_.push_back(node);
        node_map_[node->id()] = node;
        return node;
    }

    /// Create an input node
    std::shared_ptr<Node> create_input(
        const TensorSpec& spec,
        const std::string& name = ""
    ) {
        auto node = std::make_shared<Node>(next_id_++, NodeType::INPUT, name);
        node->set_output_spec(spec);
        node->set_op_type(OpType::INPUT);
        all_nodes_.push_back(node);
        node_map_[node->id()] = node;
        inputs_.push_back(node);
        return node;
    }

    /// Create a parameter node
    std::shared_ptr<Node> create_parameter(
        const TensorSpec& spec,
        const std::string& name = ""
    ) {
        auto node = std::make_shared<Node>(next_id_++, NodeType::PARAMETER, name);
        node->set_output_spec(spec);
        node->set_op_type(OpType::PARAMETER);
        all_nodes_.push_back(node);
        node_map_[node->id()] = node;
        parameters_.push_back(node);
        return node;
    }

    /// Create an operation node
    std::shared_ptr<Node> create_op(
        OpType op_type,
        const std::vector<std::shared_ptr<Node>>& inputs,
        const TensorSpec& output_spec,
        const std::string& name = ""
    ) {
        auto node = std::make_shared<Node>(next_id_++, NodeType::OPERATION, name);
        node->set_output_spec(output_spec);
        node->set_op_type(op_type);
        node->set_inputs(inputs);

        // Add as user to inputs
        for (auto& input : inputs) {
            input->add_user(node);
        }

        all_nodes_.push_back(node);
        node_map_[node->id()] = node;
        return node;
    }

    /// Get a node by ID
    std::shared_ptr<Node> node(NodeId id) const {
        auto it = node_map_.find(id);
        return it != node_map_.end() ? it->second : nullptr;
    }

    /// Mark a node as an output
    void mark_output(std::shared_ptr<Node> node) {
        if (std::find(outputs_.begin(), outputs_.end(), node) == outputs_.end()) {
            outputs_.push_back(node);
        }
    }

    // Accessors
    const std::vector<std::shared_ptr<Node>>& all_nodes() const { return all_nodes_; }
    const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }
    const std::vector<std::shared_ptr<Node>>& outputs() const { return outputs_; }
    const std::vector<std::shared_ptr<Node>>& parameters() const { return parameters_; }

    size_t num_nodes() const { return all_nodes_.size(); }

    size_t num_ops() const {
        size_t count = 0;
        for (const auto& node : all_nodes_) {
            if (node->is_operation()) count++;
        }
        return count;
    }

    /// Get nodes in topological order
    std::vector<std::shared_ptr<Node>> topological_order() const {
        std::vector<std::shared_ptr<Node>> result;
        std::set<NodeId> visited;

        std::function<void(const std::shared_ptr<Node>&)> visit =
            [&](const std::shared_ptr<Node>& node) {
                if (visited.count(node->id())) return;

                for (const auto& input : node->inputs()) {
                    visit(input);
                }

                visited.insert(node->id());
                result.push_back(node);
            };

        // Start from outputs if defined, otherwise all nodes
        if (!outputs_.empty()) {
            for (const auto& out : outputs_) {
                visit(out);
            }
        } else {
            for (const auto& node : all_nodes_) {
                visit(node);
            }
        }

        return result;
    }

    /// Estimate total memory usage
    size_t estimated_memory_bytes() const {
        size_t total = 0;
        for (const auto& node : all_nodes_) {
            total += node->output_spec().size_bytes();
        }
        return total;
    }

    /// String representation
    std::string to_string() const {
        std::ostringstream ss;
        ss << "Graph {\n";

        auto topo = topological_order();
        for (const auto& node : topo) {
            ss << "  " << node->to_string() << "\n";
        }

        if (!outputs_.empty()) {
            ss << "  outputs: [";
            for (size_t i = 0; i < outputs_.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << "%" << outputs_[i]->id();
            }
            ss << "]\n";
        }

        ss << "}";
        return ss.str();
    }

    /// Clear all nodes
    void clear() {
        all_nodes_.clear();
        node_map_.clear();
        inputs_.clear();
        outputs_.clear();
        parameters_.clear();
        next_id_ = 0;
    }

private:
    NodeId next_id_ = 0;
    std::vector<std::shared_ptr<Node>> all_nodes_;
    std::unordered_map<NodeId, std::shared_ptr<Node>> node_map_;
    std::vector<std::shared_ptr<Node>> inputs_;
    std::vector<std::shared_ptr<Node>> outputs_;
    std::vector<std::shared_ptr<Node>> parameters_;
};

}  // namespace pyflame::ir
