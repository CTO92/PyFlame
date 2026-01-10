#include "pyflame/nn/module.hpp"

#include <sstream>

namespace pyflame::nn {

// ============================================================================
// Module Implementation
// ============================================================================

std::vector<Tensor*> Module::parameters() {
    std::vector<Tensor*> result;

    // Own parameters
    for (auto& [name, param] : parameters_) {
        result.push_back(&param);
    }

    // Child module parameters
    for (auto& [name, child] : children_) {
        auto child_params = child->parameters();
        result.insert(result.end(), child_params.begin(), child_params.end());
    }

    return result;
}

std::map<std::string, Tensor*> Module::named_parameters() {
    std::map<std::string, Tensor*> result;

    // Own parameters
    for (auto& [name, param] : parameters_) {
        result[name] = &param;
    }

    // Child module parameters with prefix
    for (auto& [child_name, child] : children_) {
        auto child_params = child->named_parameters();
        for (auto& [param_name, param] : child_params) {
            result[child_name + "." + param_name] = param;
        }
    }

    return result;
}

void Module::zero_grad() {
    // Zero own parameters
    for (auto& [name, param] : parameters_) {
        auto node = param.node();
        if (node && node->has_attr("grad")) {
            node->set_attr("grad", std::shared_ptr<ir::Node>(nullptr));
        }
    }

    // Zero child parameters
    for (auto& [name, child] : children_) {
        child->zero_grad();
    }
}

void Module::train(bool mode) {
    training_ = mode;

    // Set mode for children
    for (auto& [name, child] : children_) {
        child->train(mode);
    }
}

std::map<std::string, Tensor> Module::state_dict() const {
    std::map<std::string, Tensor> result;

    // Own parameters
    for (const auto& [name, param] : parameters_) {
        result[name] = param;
    }

    // Own buffers
    for (const auto& [name, buffer] : buffers_) {
        result[name] = buffer;
    }

    // Child state dicts with prefix
    for (const auto& [child_name, child] : children_) {
        auto child_state = child->state_dict();
        for (const auto& [key, value] : child_state) {
            result[child_name + "." + key] = value;
        }
    }

    return result;
}

void Module::load_state_dict(const std::map<std::string, Tensor>& dict) {
    for (const auto& [key, value] : dict) {
        // Check if it's an own parameter
        auto param_it = parameters_.find(key);
        if (param_it != parameters_.end()) {
            param_it->second = value;
            continue;
        }

        // Check if it's an own buffer
        auto buffer_it = buffers_.find(key);
        if (buffer_it != buffers_.end()) {
            buffer_it->second = value;
            continue;
        }

        // Check if it belongs to a child
        auto dot_pos = key.find('.');
        if (dot_pos != std::string::npos) {
            std::string child_name = key.substr(0, dot_pos);
            std::string rest = key.substr(dot_pos + 1);

            auto child_it = children_.find(child_name);
            if (child_it != children_.end()) {
                std::map<std::string, Tensor> child_dict;
                child_dict[rest] = value;
                child_it->second->load_state_dict(child_dict);
            }
        }
    }
}

std::vector<std::shared_ptr<Module>> Module::children() const {
    std::vector<std::shared_ptr<Module>> result;
    for (const auto& [name, child] : children_) {
        result.push_back(child);
    }
    return result;
}

std::map<std::string, std::shared_ptr<Module>> Module::named_children() const {
    return children_;
}

std::vector<std::shared_ptr<Module>> Module::modules() {
    std::vector<std::shared_ptr<Module>> result;
    result.push_back(shared_from_this());

    for (auto& [name, child] : children_) {
        auto child_modules = child->modules();
        result.insert(result.end(), child_modules.begin(), child_modules.end());
    }

    return result;
}

std::map<std::string, std::shared_ptr<Module>> Module::named_modules() {
    std::map<std::string, std::shared_ptr<Module>> result;
    result[""] = shared_from_this();

    for (auto& [child_name, child] : children_) {
        auto child_modules = child->named_modules();
        for (auto& [name, module] : child_modules) {
            std::string full_name = child_name;
            if (!name.empty()) {
                full_name += "." + name;
            }
            result[full_name] = module;
        }
    }

    return result;
}

void Module::apply(std::function<void(Module&)> fn) {
    fn(*this);
    for (auto& [name, child] : children_) {
        child->apply(fn);
    }
}

std::string Module::to_string() const {
    std::ostringstream oss;
    oss << name_ << "(";

    bool first = true;
    for (const auto& [name, param] : parameters_) {
        if (!first) oss << ", ";
        first = false;
        oss << name << ": [";
        auto shape = param.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape[i];
        }
        oss << "]";
    }

    oss << ")";
    return oss.str();
}

Tensor& Module::register_parameter(const std::string& name, Tensor param) {
    parameters_[name] = std::move(param);
    return parameters_[name];
}

Tensor& Module::register_buffer(const std::string& name, Tensor buffer) {
    buffers_[name] = std::move(buffer);
    return buffers_[name];
}

// ============================================================================
// Sequential Implementation
// ============================================================================

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    for (auto& module : modules) {
        add(module);
    }
}

void Sequential::add(std::shared_ptr<Module> module) {
    std::string name = std::to_string(modules_.size());
    register_module(name, module);
    modules_.push_back(module);
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& module : modules_) {
        x = module->forward(x);
    }
    return x;
}

std::shared_ptr<Module> Sequential::operator[](size_t idx) const {
    return modules_.at(idx);
}

// ============================================================================
// ModuleList Implementation
// ============================================================================

void ModuleList::append(std::shared_ptr<Module> module) {
    std::string name = std::to_string(modules_.size());
    register_module(name, module);
    modules_.push_back(module);
}

std::shared_ptr<Module> ModuleList::operator[](size_t idx) const {
    return modules_.at(idx);
}

}  // namespace pyflame::nn
