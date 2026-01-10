#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <any>
#include <optional>
#include <typeinfo>

#include "pyflame/core/dtype.hpp"
#include "pyflame/core/layout.hpp"
#include "pyflame/core/safe_math.hpp"
#include "pyflame/ir/op_type.hpp"

namespace pyflame::ir {

/// Unique identifier for graph nodes
using NodeId = uint64_t;

/// Types of nodes in the computation graph
enum class NodeType : uint8_t {
    CONSTANT,       // Immutable data
    INPUT,          // Runtime input from host
    PARAMETER,      // Trainable parameter
    OPERATION,      // Result of computation
};

/// Represents the specification of a tensor
struct TensorSpec {
    std::vector<int64_t> shape;
    DType dtype = DType::Float32;
    MeshLayout layout;

    TensorSpec() = default;

    TensorSpec(std::vector<int64_t> s, DType d = DType::Float32, MeshLayout l = MeshLayout::SinglePE())
        : shape(std::move(s)), dtype(d), layout(l) {}

    /// Total number of elements (with overflow checking)
    int64_t numel() const {
        return safe_numel(shape);
    }

    /// Size in bytes (with overflow checking)
    size_t size_bytes() const {
        return safe_size_bytes(numel(), dtype_size(dtype));
    }

    /// Number of dimensions
    int ndim() const {
        return static_cast<int>(shape.size());
    }

    bool operator==(const TensorSpec& other) const {
        return shape == other.shape && dtype == other.dtype && layout == other.layout;
    }

    std::string to_string() const {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "], " + dtype_name(dtype);
        return s;
    }
};

// Forward declaration
class Graph;

/// A node in the computation graph
class Node : public std::enable_shared_from_this<Node> {
public:
    Node(NodeId id, NodeType type, const std::string& name = "")
        : id_(id), type_(type), name_(name) {}

    virtual ~Node() = default;

    // Accessors
    NodeId id() const { return id_; }
    NodeType type() const { return type_; }
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    // Output specification
    const TensorSpec& output_spec() const { return output_spec_; }
    void set_output_spec(const TensorSpec& spec) { output_spec_ = spec; }

    // Shape convenience methods
    const std::vector<int64_t>& shape() const { return output_spec_.shape; }
    DType dtype() const { return output_spec_.dtype; }
    const MeshLayout& layout() const { return output_spec_.layout; }
    int64_t numel() const { return output_spec_.numel(); }
    int ndim() const { return output_spec_.ndim(); }

    // Input edges (dependencies)
    const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }
    void add_input(std::shared_ptr<Node> input) {
        inputs_.push_back(input);
        input->add_user(shared_from_this());
    }
    void set_inputs(std::vector<std::shared_ptr<Node>> inputs) {
        inputs_ = std::move(inputs);
    }

    // Users of this node's output
    const std::vector<std::weak_ptr<Node>>& users() const { return users_; }
    void add_user(std::shared_ptr<Node> user) {
        users_.push_back(user);
    }

    // For operation nodes: the specific operation
    OpType op_type() const { return op_type_; }
    void set_op_type(OpType op) { op_type_ = op; }

    // Operation attributes (for ops with extra parameters like dim, keepdim, etc.)
    void set_attr(const std::string& key, std::any value) {
        attrs_[key] = std::move(value);
    }

    template<typename T>
    T get_attr(const std::string& key) const {
        auto it = attrs_.find(key);
        if (it == attrs_.end()) {
            throw std::runtime_error("Attribute not found: " + key);
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error(
                "Type mismatch for attribute '" + key + "': expected " +
                typeid(T).name() + ", got " + it->second.type().name());
        }
    }

    template<typename T>
    T get_attr(const std::string& key, const T& default_value) const {
        auto it = attrs_.find(key);
        if (it == attrs_.end()) {
            return default_value;
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error(
                "Type mismatch for attribute '" + key + "': expected " +
                typeid(T).name() + ", got " + it->second.type().name());
        }
    }

    /// Check if attribute exists and has the expected type
    template<typename T>
    bool has_attr_of_type(const std::string& key) const {
        auto it = attrs_.find(key);
        if (it == attrs_.end()) {
            return false;
        }
        return it->second.type() == typeid(T);
    }

    bool has_attr(const std::string& key) const {
        return attrs_.find(key) != attrs_.end();
    }

    // Check node properties
    bool is_constant() const { return type_ == NodeType::CONSTANT; }
    bool is_input() const { return type_ == NodeType::INPUT; }
    bool is_parameter() const { return type_ == NodeType::PARAMETER; }
    bool is_operation() const { return type_ == NodeType::OPERATION; }
    bool is_leaf() const { return inputs_.empty(); }

    // Constant data storage
    void set_constant_data(std::vector<uint8_t> data) {
        constant_data_ = std::move(data);
    }
    const std::vector<uint8_t>& constant_data() const { return constant_data_; }
    bool has_constant_data() const { return !constant_data_.empty(); }

    template<typename T>
    const T* constant_data_as() const {
        return reinterpret_cast<const T*>(constant_data_.data());
    }

    // Evaluation state
    bool is_evaluated() const { return evaluated_; }
    void mark_evaluated() { evaluated_ = true; }
    void clear_evaluated() { evaluated_ = false; }

    // String representation
    std::string to_string() const {
        std::string s = "%" + std::to_string(id_);
        if (!name_.empty()) {
            s += " (" + name_ + ")";
        }
        s += " = ";
        if (is_operation()) {
            s += op_type_name(op_type_);
            s += "(";
            for (size_t i = 0; i < inputs_.size(); ++i) {
                if (i > 0) s += ", ";
                s += "%" + std::to_string(inputs_[i]->id());
            }
            s += ")";
        } else {
            switch (type_) {
                case NodeType::CONSTANT: s += "constant"; break;
                case NodeType::INPUT: s += "input"; break;
                case NodeType::PARAMETER: s += "parameter"; break;
                default: s += "unknown"; break;
            }
        }
        s += " : " + output_spec_.to_string();
        return s;
    }

protected:
    NodeId id_;
    NodeType type_;
    std::string name_;
    TensorSpec output_spec_;
    std::vector<std::shared_ptr<Node>> inputs_;
    std::vector<std::weak_ptr<Node>> users_;
    OpType op_type_ = OpType::NONE;
    std::map<std::string, std::any> attrs_;
    std::vector<uint8_t> constant_data_;
    bool evaluated_ = false;
};

}  // namespace pyflame::ir
