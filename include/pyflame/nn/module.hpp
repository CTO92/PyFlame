#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

#include "pyflame/core/tensor.hpp"

namespace pyflame::nn {

/// Base class for all neural network modules
/// Provides parameter registration, state dict, and training mode
class Module : public std::enable_shared_from_this<Module> {
public:
    Module() = default;
    virtual ~Module() = default;

    // Disable copy, allow move
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;

    /// Forward pass - must be implemented by subclasses
    virtual Tensor forward(const Tensor& input) = 0;

    /// Call operator - convenience wrapper for forward
    Tensor operator()(const Tensor& input) {
        return forward(input);
    }

    /// Multi-input forward (for modules like attention)
    virtual Tensor forward(const std::vector<Tensor>& inputs) {
        if (inputs.size() != 1) {
            throw std::runtime_error("Module expects single input");
        }
        return forward(inputs[0]);
    }

    // ========================================================================
    // Parameter Management
    // ========================================================================

    /// Get all parameters (trainable tensors)
    std::vector<Tensor*> parameters();

    /// Get all named parameters
    std::map<std::string, Tensor*> named_parameters();

    /// Zero gradients of all parameters
    void zero_grad();

    // ========================================================================
    // Training Mode
    // ========================================================================

    /// Set training mode
    void train(bool mode = true);

    /// Set evaluation mode
    void eval() { train(false); }

    /// Check if in training mode
    bool is_training() const { return training_; }

    // ========================================================================
    // State Dict (for serialization)
    // ========================================================================

    /// Get state dictionary (parameter name -> tensor)
    std::map<std::string, Tensor> state_dict() const;

    /// Load state dictionary
    void load_state_dict(const std::map<std::string, Tensor>& dict);

    // ========================================================================
    // Module Hierarchy
    // ========================================================================

    /// Get all child modules
    std::vector<std::shared_ptr<Module>> children() const;

    /// Get all named child modules
    std::map<std::string, std::shared_ptr<Module>> named_children() const;

    /// Get all modules (recursive)
    std::vector<std::shared_ptr<Module>> modules();

    /// Get all named modules (recursive)
    std::map<std::string, std::shared_ptr<Module>> named_modules();

    /// Apply a function to all modules
    void apply(std::function<void(Module&)> fn);

    // ========================================================================
    // String Representation
    // ========================================================================

    /// Get module name
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    /// String representation
    virtual std::string to_string() const;

protected:
    /// Register a parameter tensor
    Tensor& register_parameter(const std::string& name, Tensor param);

    /// Register a child module
    template<typename T>
    std::shared_ptr<T> register_module(const std::string& name, std::shared_ptr<T> module) {
        static_assert(std::is_base_of_v<Module, T>, "T must derive from Module");
        module->set_name(name);
        children_[name] = module;
        return module;
    }

    /// Register a buffer (non-trainable tensor)
    Tensor& register_buffer(const std::string& name, Tensor buffer);

private:
    std::string name_;
    bool training_ = true;

    std::map<std::string, Tensor> parameters_;
    std::map<std::string, Tensor> buffers_;
    std::map<std::string, std::shared_ptr<Module>> children_;
};

/// Sequential container - applies modules in order
class Sequential : public Module {
public:
    Sequential() = default;

    /// Construct from list of modules
    Sequential(std::initializer_list<std::shared_ptr<Module>> modules);

    /// Add a module
    void add(std::shared_ptr<Module> module);

    /// Forward pass
    Tensor forward(const Tensor& input) override;

    /// Access module by index
    std::shared_ptr<Module> operator[](size_t idx) const;

    /// Number of modules
    size_t size() const { return modules_.size(); }

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

/// ModuleList - holds modules in a list
class ModuleList : public Module {
public:
    ModuleList() = default;

    /// Add a module
    void append(std::shared_ptr<Module> module);

    /// Access module
    std::shared_ptr<Module> operator[](size_t idx) const;

    /// Forward - not typically called directly
    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("ModuleList does not implement forward()");
    }

    size_t size() const { return modules_.size(); }

    auto begin() { return modules_.begin(); }
    auto end() { return modules_.end(); }
    auto begin() const { return modules_.begin(); }
    auto end() const { return modules_.end(); }

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace pyflame::nn
