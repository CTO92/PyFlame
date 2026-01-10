# Phase 2: ML Primitives - Implementation Plan

**PyFlame Version:** Pre-Release Alpha 1.0
**Phase:** 2 of 4
**Focus:** ML Primitives
**Target Duration:** Months 6-12

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. Plans described here are subject to change.

---

## Table of Contents

1. [Phase 2 Overview](#1-phase-2-overview)
2. [Automatic Differentiation (Autograd)](#2-automatic-differentiation-autograd)
3. [Neural Network Layers](#3-neural-network-layers)
4. [Loss Functions](#4-loss-functions)
5. [Optimizers](#5-optimizers)
6. [Training Infrastructure](#6-training-infrastructure)
7. [CSL Backend Extensions](#7-csl-backend-extensions)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Technical Decisions](#9-technical-decisions)

---

## 1. Phase 2 Overview

### 1.1 Goals

Phase 2 builds upon Phase 1's core infrastructure to add the primitives necessary for training neural networks:

| Component | Description |
|-----------|-------------|
| **Autograd** | Automatic computation of gradients for backpropagation |
| **NN Layers** | Convolutions, normalization, pooling, attention |
| **Loss Functions** | Cross-entropy, MSE, and other training objectives |
| **Optimizers** | SGD, Adam, and learning rate scheduling |
| **Training Loop** | Forward pass, backward pass, parameter updates |

### 1.2 Dependencies on Phase 1

Phase 2 requires these Phase 1 components to be complete:

- [x] Core tensor class with lazy evaluation
- [x] Computation graph (IR) system
- [x] Shape inference
- [x] Elementwise operations
- [x] Reduction operations
- [x] Matrix multiplication
- [x] CSL code generation framework
- [x] Python bindings
- [x] CPU reference implementation

### 1.3 Key Design Principle: Gradient-Aware IR

The central architectural decision for Phase 2 is extending the IR to be gradient-aware. Each operation node must know how to compute its backward pass.

---

## 2. Automatic Differentiation (Autograd)

### 2.1 Overview

Autograd enables automatic computation of gradients, essential for training neural networks via backpropagation.

### 2.2 Design Approach: Reverse-Mode Autodiff

PyFlame will use **reverse-mode automatic differentiation** (backpropagation), which is efficient for neural networks where the number of outputs (typically 1 loss scalar) is much smaller than the number of inputs (model parameters).

```
Forward Pass:  x → f(x) → y → g(y) → z → loss
Backward Pass: ∂loss/∂z → ∂loss/∂y → ∂loss/∂x
```

### 2.3 Implementation Strategy

#### Step 1: Extend Node to Track Gradient Information

```cpp
// include/pyflame/ir/node.hpp - Extend existing Node class

namespace pyflame::ir {

// Gradient function signature
// Takes: output gradient, forward inputs, forward output
// Returns: gradients with respect to each input
using GradientFn = std::function<std::vector<std::shared_ptr<Node>>(
    std::shared_ptr<Node> grad_output,
    const std::vector<std::shared_ptr<Node>>& inputs,
    std::shared_ptr<Node> output
)>;

class Node {
    // ... existing members ...

    // NEW: Gradient computation
    GradientFn grad_fn_;           // Function to compute input gradients
    bool requires_grad_ = false;    // Whether to track gradients
    std::shared_ptr<Node> grad_;    // Accumulated gradient (for leaves)

public:
    void set_requires_grad(bool requires);
    bool requires_grad() const;

    // Backward pass
    void backward();
    std::shared_ptr<Node> grad() const;
    void zero_grad();
};

}  // namespace pyflame::ir
```

#### Step 2: Register Gradient Functions for Each Op

Each operation needs a corresponding gradient rule:

```cpp
// src/ir/gradients.cpp

namespace pyflame::ir {

// Registry of gradient functions
std::unordered_map<OpType, GradientFn> gradient_registry;

void register_gradients() {
    // Addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
    gradient_registry[OpType::Add] = [](auto grad_out, auto inputs, auto) {
        return {grad_out, grad_out};  // Pass gradient unchanged to both inputs
    };

    // Multiplication: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
    gradient_registry[OpType::Mul] = [](auto grad_out, auto inputs, auto) {
        auto grad_a = make_node(OpType::Mul, {grad_out, inputs[1]});
        auto grad_b = make_node(OpType::Mul, {grad_out, inputs[0]});
        return {grad_a, grad_b};
    };

    // MatMul: ∂(A@B)/∂A = grad @ B^T, ∂(A@B)/∂B = A^T @ grad
    gradient_registry[OpType::MatMul] = [](auto grad_out, auto inputs, auto) {
        auto A = inputs[0];
        auto B = inputs[1];
        auto grad_A = make_node(OpType::MatMul, {grad_out, transpose(B)});
        auto grad_B = make_node(OpType::MatMul, {transpose(A), grad_out});
        return {grad_A, grad_B};
    };

    // ReLU: ∂relu(x)/∂x = 1 if x > 0, else 0
    gradient_registry[OpType::ReLU] = [](auto grad_out, auto inputs, auto output) {
        auto mask = make_node(OpType::Greater, {inputs[0], zeros_like(inputs[0])});
        return {make_node(OpType::Mul, {grad_out, mask})};
    };

    // Sum: broadcast gradient back
    gradient_registry[OpType::Sum] = [](auto grad_out, auto inputs, auto) {
        return {broadcast_to(grad_out, inputs[0]->shape())};
    };

    // ... more gradient rules ...
}

}  // namespace pyflame::ir
```

#### Step 3: Backward Pass Algorithm

```cpp
// src/ir/backward.cpp

void Node::backward() {
    if (!requires_grad_) return;

    // For scalar outputs, start with gradient of 1
    if (grad_ == nullptr) {
        grad_ = ones_like(shared_from_this());
    }

    // Topological sort (reverse order)
    std::vector<std::shared_ptr<Node>> topo_order;
    std::unordered_set<Node*> visited;

    std::function<void(std::shared_ptr<Node>)> topo_sort =
        [&](std::shared_ptr<Node> node) {
            if (visited.count(node.get())) return;
            visited.insert(node.get());
            for (auto& input : node->inputs()) {
                if (input->requires_grad()) {
                    topo_sort(input);
                }
            }
            topo_order.push_back(node);
        };

    topo_sort(shared_from_this());

    // Process in reverse topological order
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        auto node = *it;
        if (node->grad_fn_) {
            auto input_grads = node->grad_fn_(node->grad_, node->inputs(), node);

            // Accumulate gradients to inputs
            for (size_t i = 0; i < input_grads.size(); ++i) {
                auto& input = node->inputs()[i];
                if (input->requires_grad()) {
                    if (input->grad_ == nullptr) {
                        input->grad_ = input_grads[i];
                    } else {
                        input->grad_ = make_node(OpType::Add,
                            {input->grad_, input_grads[i]});
                    }
                }
            }
        }
    }
}
```

#### Step 4: Python API for Autograd

```python
# python/pyflame/autograd.py

def backward(tensor, grad_output=None):
    """Compute gradients via backpropagation.

    Args:
        tensor: Output tensor (typically loss)
        grad_output: Gradient of some downstream scalar with respect to tensor.
                    If None, assumes tensor is scalar and uses 1.0.
    """
    tensor._backward(grad_output)

def grad(outputs, inputs, grad_outputs=None, retain_graph=False):
    """Compute gradients of outputs with respect to inputs.

    Args:
        outputs: Tensor or sequence of tensors
        inputs: Tensor or sequence of tensors
        grad_outputs: Gradients w.r.t. each output
        retain_graph: If False, free the graph after computing gradients

    Returns:
        Tuple of gradients with respect to each input
    """
    # Implementation
    pass
```

### 2.4 Gradient Rules Reference

| Operation | Gradient Rule |
|-----------|---------------|
| `y = a + b` | `∂y/∂a = 1`, `∂y/∂b = 1` |
| `y = a - b` | `∂y/∂a = 1`, `∂y/∂b = -1` |
| `y = a * b` | `∂y/∂a = b`, `∂y/∂b = a` |
| `y = a / b` | `∂y/∂a = 1/b`, `∂y/∂b = -a/b²` |
| `y = A @ B` | `∂y/∂A = grad @ Bᵀ`, `∂y/∂B = Aᵀ @ grad` |
| `y = relu(x)` | `∂y/∂x = grad * (x > 0)` |
| `y = sigmoid(x)` | `∂y/∂x = grad * y * (1 - y)` |
| `y = tanh(x)` | `∂y/∂x = grad * (1 - y²)` |
| `y = exp(x)` | `∂y/∂x = grad * y` |
| `y = log(x)` | `∂y/∂x = grad / x` |
| `y = sum(x)` | `∂y/∂x = broadcast(grad)` |
| `y = mean(x)` | `∂y/∂x = broadcast(grad) / n` |
| `y = softmax(x)` | `∂y/∂x = y * (grad - sum(grad * y))` |

---

## 3. Neural Network Layers

### 3.1 Layer Base Class

```cpp
// include/pyflame/nn/module.hpp

namespace pyflame::nn {

class Module {
public:
    virtual ~Module() = default;

    // Forward pass
    virtual Tensor forward(const Tensor& input) = 0;

    // Convenience: call operator
    Tensor operator()(const Tensor& input) { return forward(input); }

    // Parameters
    virtual std::vector<Tensor*> parameters();
    virtual void zero_grad();

    // Training mode
    void train(bool mode = true);
    void eval() { train(false); }
    bool is_training() const { return training_; }

    // State dict for serialization
    virtual std::map<std::string, Tensor> state_dict() const;
    virtual void load_state_dict(const std::map<std::string, Tensor>& dict);

protected:
    bool training_ = true;
    std::vector<std::pair<std::string, Tensor>> named_parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> named_modules_;

    // Register a parameter
    Tensor& register_parameter(const std::string& name, Tensor param);

    // Register a submodule
    template<typename T>
    T& register_module(const std::string& name, T module);
};

}  // namespace pyflame::nn
```

### 3.2 Linear Layer

```cpp
// include/pyflame/nn/linear.hpp

namespace pyflame::nn {

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true);

    Tensor forward(const Tensor& input) override;

    // Parameters
    Tensor weight;  // [out_features, in_features]
    Tensor bias;    // [out_features] or empty if no bias

private:
    int64_t in_features_;
    int64_t out_features_;
    bool use_bias_;
};

// Implementation
Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias) {

    // Kaiming initialization
    float std = std::sqrt(2.0f / in_features);
    weight = register_parameter("weight",
        Tensor::randn({out_features, in_features}) * std);
    weight.set_requires_grad(true);

    if (use_bias_) {
        bias = register_parameter("bias", Tensor::zeros({out_features}));
        bias.set_requires_grad(true);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // input: [batch, in_features]
    // weight: [out_features, in_features]
    // output: [batch, out_features]
    auto output = matmul(input, weight.transpose(0, 1));
    if (use_bias_) {
        output = output + bias;
    }
    return output;
}

}  // namespace pyflame::nn
```

### 3.3 Convolution Layers

```cpp
// include/pyflame/nn/conv.hpp

namespace pyflame::nn {

class Conv2d : public Module {
public:
    Conv2d(int64_t in_channels, int64_t out_channels,
           std::vector<int64_t> kernel_size,
           std::vector<int64_t> stride = {1, 1},
           std::vector<int64_t> padding = {0, 0},
           std::vector<int64_t> dilation = {1, 1},
           int64_t groups = 1,
           bool bias = true);

    Tensor forward(const Tensor& input) override;

    Tensor weight;  // [out_channels, in_channels/groups, kH, kW]
    Tensor bias;    // [out_channels]

private:
    int64_t in_channels_, out_channels_;
    std::vector<int64_t> kernel_size_, stride_, padding_, dilation_;
    int64_t groups_;
    bool use_bias_;
};

}  // namespace pyflame::nn
```

**Convolution Implementation Strategy:**

For Cerebras WSE, convolution can be implemented efficiently using:

1. **im2col + MatMul**: Transform convolution into matrix multiplication
   - Unfold input patches into columns
   - Reshape kernel into a matrix
   - Perform matmul
   - Reshape back

2. **Direct convolution on PE mesh**: Distribute spatial dimensions across PEs
   - Each PE handles a spatial tile
   - Wavelets communicate halo regions

### 3.4 Normalization Layers

```cpp
// include/pyflame/nn/normalization.hpp

namespace pyflame::nn {

class BatchNorm2d : public Module {
public:
    BatchNorm2d(int64_t num_features,
                float eps = 1e-5,
                float momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true);

    Tensor forward(const Tensor& input) override;

    Tensor weight;           // [num_features] - gamma
    Tensor bias;             // [num_features] - beta
    Tensor running_mean;     // [num_features]
    Tensor running_var;      // [num_features]
    int64_t num_batches_tracked;

private:
    int64_t num_features_;
    float eps_, momentum_;
    bool affine_, track_running_stats_;
};

class LayerNorm : public Module {
public:
    LayerNorm(std::vector<int64_t> normalized_shape,
              float eps = 1e-5,
              bool elementwise_affine = true);

    Tensor forward(const Tensor& input) override;

    Tensor weight;  // gamma
    Tensor bias;    // beta

private:
    std::vector<int64_t> normalized_shape_;
    float eps_;
    bool elementwise_affine_;
};

}  // namespace pyflame::nn
```

### 3.5 Pooling Layers

```cpp
// include/pyflame/nn/pooling.hpp

namespace pyflame::nn {

class MaxPool2d : public Module {
public:
    MaxPool2d(std::vector<int64_t> kernel_size,
              std::vector<int64_t> stride = {},
              std::vector<int64_t> padding = {0, 0});

    Tensor forward(const Tensor& input) override;

private:
    std::vector<int64_t> kernel_size_, stride_, padding_;
};

class AvgPool2d : public Module {
public:
    AvgPool2d(std::vector<int64_t> kernel_size,
              std::vector<int64_t> stride = {},
              std::vector<int64_t> padding = {0, 0});

    Tensor forward(const Tensor& input) override;

private:
    std::vector<int64_t> kernel_size_, stride_, padding_;
};

class AdaptiveAvgPool2d : public Module {
public:
    AdaptiveAvgPool2d(std::vector<int64_t> output_size);

    Tensor forward(const Tensor& input) override;

private:
    std::vector<int64_t> output_size_;
};

}  // namespace pyflame::nn
```

### 3.6 Dropout

```cpp
// include/pyflame/nn/dropout.hpp

namespace pyflame::nn {

class Dropout : public Module {
public:
    Dropout(float p = 0.5, bool inplace = false);

    Tensor forward(const Tensor& input) override;

private:
    float p_;
    bool inplace_;
};

}  // namespace pyflame::nn

// Implementation
Tensor Dropout::forward(const Tensor& input) {
    if (!is_training() || p_ == 0.0f) {
        return input;
    }

    // Create mask: 1 with probability (1-p), 0 with probability p
    auto mask = Tensor::rand(input.shape()) > p_;

    // Scale by 1/(1-p) to maintain expected value
    float scale = 1.0f / (1.0f - p_);

    return input * mask * scale;
}
```

### 3.7 Attention Layers

```cpp
// include/pyflame/nn/attention.hpp

namespace pyflame::nn {

class MultiheadAttention : public Module {
public:
    MultiheadAttention(int64_t embed_dim,
                       int64_t num_heads,
                       float dropout = 0.0f,
                       bool bias = true,
                       bool add_bias_kv = false,
                       bool add_zero_attn = false,
                       int64_t kdim = 0,   // 0 = same as embed_dim
                       int64_t vdim = 0);  // 0 = same as embed_dim

    // Returns (output, attention_weights)
    std::pair<Tensor, Tensor> forward(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        const Tensor& attn_mask = Tensor(),
        bool need_weights = true);

    Linear in_proj_q, in_proj_k, in_proj_v;
    Linear out_proj;

private:
    int64_t embed_dim_, num_heads_, head_dim_;
    float dropout_;
    Dropout dropout_layer;
};

}  // namespace pyflame::nn
```

### 3.8 Embedding Layer

```cpp
// include/pyflame/nn/embedding.hpp

namespace pyflame::nn {

class Embedding : public Module {
public:
    Embedding(int64_t num_embeddings,
              int64_t embedding_dim,
              int64_t padding_idx = -1);  // -1 = no padding

    Tensor forward(const Tensor& indices) override;

    Tensor weight;  // [num_embeddings, embedding_dim]

private:
    int64_t num_embeddings_, embedding_dim_;
    int64_t padding_idx_;
};

}  // namespace pyflame::nn
```

---

## 4. Loss Functions

### 4.1 Loss Function Base

```cpp
// include/pyflame/nn/loss.hpp

namespace pyflame::nn {

class Loss : public Module {
public:
    enum class Reduction { None, Mean, Sum };

    Loss(Reduction reduction = Reduction::Mean);

protected:
    Reduction reduction_;

    Tensor apply_reduction(const Tensor& loss) const;
};

}  // namespace pyflame::nn
```

### 4.2 Common Loss Functions

```cpp
// include/pyflame/nn/loss.hpp

namespace pyflame::nn {

class MSELoss : public Loss {
public:
    MSELoss(Reduction reduction = Reduction::Mean);

    Tensor forward(const Tensor& input, const Tensor& target);
};

class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss(Reduction reduction = Reduction::Mean,
                     int64_t ignore_index = -100,
                     float label_smoothing = 0.0f);

    Tensor forward(const Tensor& input, const Tensor& target);

private:
    int64_t ignore_index_;
    float label_smoothing_;
};

class BCELoss : public Loss {
public:
    BCELoss(Reduction reduction = Reduction::Mean);

    Tensor forward(const Tensor& input, const Tensor& target);
};

class BCEWithLogitsLoss : public Loss {
public:
    BCEWithLogitsLoss(Reduction reduction = Reduction::Mean);

    Tensor forward(const Tensor& input, const Tensor& target);
};

class NLLLoss : public Loss {
public:
    NLLLoss(Reduction reduction = Reduction::Mean,
            int64_t ignore_index = -100);

    Tensor forward(const Tensor& input, const Tensor& target);

private:
    int64_t ignore_index_;
};

class L1Loss : public Loss {
public:
    L1Loss(Reduction reduction = Reduction::Mean);

    Tensor forward(const Tensor& input, const Tensor& target);
};

class SmoothL1Loss : public Loss {
public:
    SmoothL1Loss(Reduction reduction = Reduction::Mean, float beta = 1.0f);

    Tensor forward(const Tensor& input, const Tensor& target);

private:
    float beta_;
};

}  // namespace pyflame::nn
```

### 4.3 Loss Implementation Examples

```cpp
// src/nn/loss.cpp

Tensor MSELoss::forward(const Tensor& input, const Tensor& target) {
    auto diff = input - target;
    auto squared = diff * diff;
    return apply_reduction(squared);
}

Tensor CrossEntropyLoss::forward(const Tensor& input, const Tensor& target) {
    // input: [N, C] logits
    // target: [N] class indices

    auto log_probs = log_softmax(input, /*dim=*/1);

    // Gather log probabilities at target indices
    auto nll = -gather(log_probs, /*dim=*/1, target.unsqueeze(1)).squeeze(1);

    // Apply label smoothing if specified
    if (label_smoothing_ > 0.0f) {
        auto smooth_loss = -log_probs.mean(/*dim=*/1);
        nll = (1.0f - label_smoothing_) * nll + label_smoothing_ * smooth_loss;
    }

    return apply_reduction(nll);
}
```

---

## 5. Optimizers

### 5.1 Optimizer Base Class

```cpp
// include/pyflame/optim/optimizer.hpp

namespace pyflame::optim {

struct OptimizerOptions {
    float lr = 0.001f;
    float weight_decay = 0.0f;
};

class Optimizer {
public:
    Optimizer(std::vector<Tensor*> parameters, OptimizerOptions options);
    virtual ~Optimizer() = default;

    // Perform optimization step
    virtual void step() = 0;

    // Zero all parameter gradients
    void zero_grad();

    // State dict for checkpointing
    virtual std::map<std::string, Tensor> state_dict() const;
    virtual void load_state_dict(const std::map<std::string, Tensor>& dict);

protected:
    std::vector<Tensor*> parameters_;
    OptimizerOptions options_;
    int64_t step_count_ = 0;
};

}  // namespace pyflame::optim
```

### 5.2 SGD Optimizer

```cpp
// include/pyflame/optim/sgd.hpp

namespace pyflame::optim {

struct SGDOptions : OptimizerOptions {
    float momentum = 0.0f;
    bool nesterov = false;
    float dampening = 0.0f;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> parameters, SGDOptions options = {});

    void step() override;

private:
    SGDOptions options_;
    std::vector<Tensor> momentum_buffers_;
};

// Implementation
void SGD::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = *parameters_[i];
        auto grad = param.grad();

        if (grad.is_empty()) continue;

        // Apply weight decay
        if (options_.weight_decay != 0.0f) {
            grad = grad + param * options_.weight_decay;
        }

        // Apply momentum
        if (options_.momentum != 0.0f) {
            if (momentum_buffers_[i].is_empty()) {
                momentum_buffers_[i] = grad.clone();
            } else {
                momentum_buffers_[i] =
                    momentum_buffers_[i] * options_.momentum +
                    grad * (1.0f - options_.dampening);
            }

            if (options_.nesterov) {
                grad = grad + momentum_buffers_[i] * options_.momentum;
            } else {
                grad = momentum_buffers_[i];
            }
        }

        // Update parameter
        param.add_(grad * (-options_.lr));
    }

    step_count_++;
}

}  // namespace pyflame::optim
```

### 5.3 Adam Optimizer

```cpp
// include/pyflame/optim/adam.hpp

namespace pyflame::optim {

struct AdamOptions : OptimizerOptions {
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    bool amsgrad = false;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> parameters, AdamOptions options = {});

    void step() override;

private:
    AdamOptions options_;
    std::vector<Tensor> exp_avg_;      // First moment
    std::vector<Tensor> exp_avg_sq_;   // Second moment
    std::vector<Tensor> max_exp_avg_sq_; // For AMSGrad
};

// Implementation
void Adam::step() {
    step_count_++;

    float bias_correction1 = 1.0f - std::pow(options_.beta1, step_count_);
    float bias_correction2 = 1.0f - std::pow(options_.beta2, step_count_);

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = *parameters_[i];
        auto grad = param.grad();

        if (grad.is_empty()) continue;

        // Apply weight decay (AdamW style)
        if (options_.weight_decay != 0.0f) {
            param.add_(param * (-options_.lr * options_.weight_decay));
        }

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * options_.beta1 + grad * (1.0f - options_.beta1);

        // Update biased second moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * options_.beta2 +
                         (grad * grad) * (1.0f - options_.beta2);

        Tensor denom;
        if (options_.amsgrad) {
            max_exp_avg_sq_[i] = max(max_exp_avg_sq_[i], exp_avg_sq_[i]);
            denom = sqrt(max_exp_avg_sq_[i]) / std::sqrt(bias_correction2) +
                    options_.eps;
        } else {
            denom = sqrt(exp_avg_sq_[i]) / std::sqrt(bias_correction2) +
                    options_.eps;
        }

        float step_size = options_.lr / bias_correction1;
        param.add_((exp_avg_[i] / denom) * (-step_size));
    }
}

}  // namespace pyflame::optim
```

### 5.4 Learning Rate Schedulers

```cpp
// include/pyflame/optim/lr_scheduler.hpp

namespace pyflame::optim {

class LRScheduler {
public:
    LRScheduler(Optimizer& optimizer);
    virtual ~LRScheduler() = default;

    virtual void step() = 0;
    float get_last_lr() const { return last_lr_; }

protected:
    Optimizer& optimizer_;
    float base_lr_;
    float last_lr_;
    int64_t last_epoch_ = 0;

    void set_lr(float lr);
};

class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, int64_t step_size, float gamma = 0.1f);
    void step() override;

private:
    int64_t step_size_;
    float gamma_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int64_t T_max, float eta_min = 0.0f);
    void step() override;

private:
    int64_t T_max_;
    float eta_min_;
};

class LinearLR : public LRScheduler {
public:
    LinearLR(Optimizer& optimizer,
             float start_factor = 1.0f/3.0f,
             float end_factor = 1.0f,
             int64_t total_iters = 5);
    void step() override;

private:
    float start_factor_, end_factor_;
    int64_t total_iters_;
};

}  // namespace pyflame::optim
```

---

## 6. Training Infrastructure

### 6.1 Gradient Context Manager

```cpp
// include/pyflame/autograd/grad_mode.hpp

namespace pyflame::autograd {

// Global gradient computation state
class GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);

private:
    static thread_local bool enabled_;
};

// RAII class for temporarily disabling gradients
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

private:
    bool prev_mode_;
};

// RAII class for temporarily enabling gradients
class EnableGradGuard {
public:
    EnableGradGuard();
    ~EnableGradGuard();

private:
    bool prev_mode_;
};

}  // namespace pyflame::autograd
```

**Python API:**

```python
# python/pyflame/autograd.py

class no_grad:
    """Context manager to disable gradient computation."""
    def __enter__(self):
        self.prev = _pyflame_cpp.GradMode.is_enabled()
        _pyflame_cpp.GradMode.set_enabled(False)
        return self

    def __exit__(self, *args):
        _pyflame_cpp.GradMode.set_enabled(self.prev)

class enable_grad:
    """Context manager to enable gradient computation."""
    def __enter__(self):
        self.prev = _pyflame_cpp.GradMode.is_enabled()
        _pyflame_cpp.GradMode.set_enabled(True)
        return self

    def __exit__(self, *args):
        _pyflame_cpp.GradMode.set_enabled(self.prev)
```

### 6.2 Training Loop Pattern

```python
# examples/python/training_loop.py

import pyflame as pf
import pyflame.nn as nn
import pyflame.optim as optim

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = pf.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model, loss, optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert to PyFlame tensors
        data = pf.from_numpy(data)
        target = pf.from_numpy(target)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Evaluate (triggers computation)
        pf.eval(loss)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.numpy():.4f}")

    # Validation
    with pf.no_grad():
        model.eval()
        # ... validation code ...
        model.train()
```

### 6.3 Checkpointing

```cpp
// include/pyflame/utils/checkpoint.hpp

namespace pyflame::utils {

void save(const std::map<std::string, Tensor>& state_dict,
          const std::string& path);

std::map<std::string, Tensor> load(const std::string& path);

// For entire training state
struct TrainingState {
    std::map<std::string, Tensor> model_state;
    std::map<std::string, Tensor> optimizer_state;
    int64_t epoch;
    int64_t global_step;
};

void save_checkpoint(const TrainingState& state, const std::string& path);
TrainingState load_checkpoint(const std::string& path);

}  // namespace pyflame::utils
```

---

## 7. CSL Backend Extensions

### 7.1 Gradient Kernels

For each forward operation, we need corresponding backward CSL kernels:

```csl
// templates/backward/matmul_backward.csl

// Compute gradient with respect to A in C = A @ B
// grad_A = grad_C @ B^T
task compute_grad_A(color: color, pe_id: PECoord) void {
    // Receive gradient from output
    const grad_C = @wavelet.recv(color);

    // Load local B tile (transposed)
    const B_T = load_tile_transposed();

    // Compute local contribution to grad_A
    const local_grad_A = matmul_tile(grad_C, B_T);

    // Reduce across PEs if needed
    @wavelet.send(reduce_color, local_grad_A);
}
```

### 7.2 Memory-Efficient Gradients

For large activations, use gradient checkpointing:

```cpp
// include/pyflame/autograd/checkpoint.hpp

namespace pyflame::autograd {

// Wrapper that doesn't save activations for backward
// Instead, recomputes forward during backward
template<typename Fn>
Tensor checkpoint(Fn forward_fn, const Tensor& input);

}  // namespace pyflame::autograd
```

### 7.3 CSL Templates for New Operations

Each new operation needs CSL code generation templates:

```cpp
// src/backend/csl_templates.cpp

void CSLTemplates::register_phase2_templates() {
    // Convolution templates
    templates_[OpType::Conv2d] = R"(
        // Conv2d kernel using im2col approach
        task conv2d_forward(/* ... */) {
            // Implementation
        }
    )";

    // BatchNorm template
    templates_[OpType::BatchNorm] = R"(
        // Batch normalization kernel
        task batchnorm_forward(/* ... */) {
            // Mean and variance computation with reduction
            // Normalization with learned parameters
        }
    )";

    // Pooling templates
    templates_[OpType::MaxPool2d] = R"(
        task maxpool2d_forward(/* ... */) {
            // Max pooling with index tracking for backward
        }
    )";
}
```

---

## 8. Implementation Roadmap

### 8.1 Milestones

| Milestone | Deliverable | Duration |
|-----------|-------------|----------|
| M2.1 | Autograd infrastructure (Node extensions, backward pass) | Weeks 1-4 |
| M2.2 | Gradient rules for all Phase 1 operations | Weeks 5-6 |
| M2.3 | Linear layer with working training | Weeks 7-8 |
| M2.4 | Loss functions (MSE, CrossEntropy) | Weeks 9-10 |
| M2.5 | SGD and Adam optimizers | Weeks 11-12 |
| M2.6 | Conv2d (forward and backward) | Weeks 13-16 |
| M2.7 | BatchNorm, LayerNorm | Weeks 17-18 |
| M2.8 | Pooling layers | Weeks 19-20 |
| M2.9 | Dropout, Attention | Weeks 21-22 |
| M2.10 | CSL kernels for all new ops | Weeks 23-26 |

### 8.2 Testing Strategy

1. **Gradient Checking**: Numerical gradient verification for all ops
2. **Unit Tests**: Each layer, loss, optimizer tested independently
3. **Integration Tests**: Full training loops on simple datasets
4. **CSL Simulation**: All kernels tested on Cerebras simulator
5. **Comparison Tests**: Match PyTorch results within tolerance

### 8.3 Dependencies Between Components

```
                    Autograd
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
      Linear        Conv2d       Attention
         │             │             │
         └──────┬──────┴──────┬──────┘
                ▼             ▼
           Loss Fns      Optimizers
                │             │
                └──────┬──────┘
                       ▼
                Training Loop
                       │
                       ▼
              CSL Code Generation
```

---

## 9. Technical Decisions

### 9.1 Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Autodiff Mode | Reverse-mode | Efficient for NN training (many inputs, scalar output) |
| Gradient Storage | Accumulate in Node | Simpler memory model, works with lazy eval |
| Parameter Init | Kaiming by default | Standard for ReLU networks |
| Optimizer Style | In-place updates | Memory efficient |
| Gradient Checkpointing | Optional, per-layer | Flexibility for memory vs. compute tradeoff |

### 9.2 Open Questions

1. **Mixed Precision Training**: Support for fp16/bf16 training with loss scaling?
2. **Distributed Training**: Multi-chip gradient synchronization?
3. **Dynamic Batching**: How to handle variable sequence lengths?
4. **JIT Compilation**: Cache compiled CSL for repeated forward/backward?

### 9.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Complex gradient rules for WSE ops | Medium | High | Extensive testing, numerical gradient checks |
| Memory pressure from storing activations | High | Medium | Gradient checkpointing, lazy materialization |
| CSL backward kernel complexity | Medium | High | Template-based generation, thorough simulation |
| Performance regression vs. Phase 1 | Low | Medium | Continuous benchmarking |

---

## Appendix A: Complete Gradient Rules

### A.1 Elementwise Operations

```cpp
// Add
gradient_registry[OpType::Add] = [](grad_out, inputs, _) {
    // Both inputs receive the full gradient
    auto grad_a = grad_out;
    auto grad_b = grad_out;
    // Handle broadcasting by summing over broadcast dimensions
    return {reduce_to_shape(grad_a, inputs[0]->shape()),
            reduce_to_shape(grad_b, inputs[1]->shape())};
};

// Sub
gradient_registry[OpType::Sub] = [](grad_out, inputs, _) {
    return {reduce_to_shape(grad_out, inputs[0]->shape()),
            reduce_to_shape(-grad_out, inputs[1]->shape())};
};

// Mul
gradient_registry[OpType::Mul] = [](grad_out, inputs, _) {
    auto grad_a = grad_out * inputs[1];
    auto grad_b = grad_out * inputs[0];
    return {reduce_to_shape(grad_a, inputs[0]->shape()),
            reduce_to_shape(grad_b, inputs[1]->shape())};
};

// Div
gradient_registry[OpType::Div] = [](grad_out, inputs, _) {
    auto a = inputs[0], b = inputs[1];
    auto grad_a = grad_out / b;
    auto grad_b = -grad_out * a / (b * b);
    return {reduce_to_shape(grad_a, a->shape()),
            reduce_to_shape(grad_b, b->shape())};
};
```

### A.2 Activation Functions

```cpp
// Sigmoid: y = 1/(1+exp(-x)), dy/dx = y*(1-y)
gradient_registry[OpType::Sigmoid] = [](grad_out, inputs, output) {
    auto y = output;
    return {grad_out * y * (ones_like(y) - y)};
};

// Tanh: y = tanh(x), dy/dx = 1 - y^2
gradient_registry[OpType::Tanh] = [](grad_out, inputs, output) {
    auto y = output;
    return {grad_out * (ones_like(y) - y * y)};
};

// GELU (approximate): y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
gradient_registry[OpType::GELU] = [](grad_out, inputs, output) {
    // Complex gradient - see paper
    auto x = inputs[0];
    // ... implementation ...
};

// Softmax: complex gradient involving Jacobian
gradient_registry[OpType::Softmax] = [](grad_out, inputs, output) {
    auto y = output;  // softmax output
    auto sum_term = (grad_out * y).sum(/*dim=*/-1, /*keepdim=*/true);
    return {y * (grad_out - sum_term)};
};
```

### A.3 Matrix Operations

```cpp
// MatMul: C = A @ B
// dL/dA = dL/dC @ B^T
// dL/dB = A^T @ dL/dC
gradient_registry[OpType::MatMul] = [](grad_out, inputs, _) {
    auto A = inputs[0], B = inputs[1];
    auto grad_A = matmul(grad_out, transpose(B, -2, -1));
    auto grad_B = matmul(transpose(A, -2, -1), grad_out);
    return {grad_A, grad_B};
};

// Transpose
gradient_registry[OpType::Transpose] = [](grad_out, inputs, output) {
    int dim0 = output->attr<int>("dim0");
    int dim1 = output->attr<int>("dim1");
    return {transpose(grad_out, dim0, dim1)};
};
```

### A.4 Reduction Operations

```cpp
// Sum
gradient_registry[OpType::Sum] = [](grad_out, inputs, output) {
    auto input_shape = inputs[0]->shape();
    int dim = output->attr<int>("dim", -1);
    bool keepdim = output->attr<bool>("keepdim", false);

    if (dim < 0) {
        // Global sum: broadcast scalar gradient
        return {broadcast_to(grad_out, input_shape)};
    } else {
        // Sum along dimension: unsqueeze if needed, then broadcast
        auto grad = keepdim ? grad_out : unsqueeze(grad_out, dim);
        return {broadcast_to(grad, input_shape)};
    }
};

// Mean: same as sum, but divide by count
gradient_registry[OpType::Mean] = [](grad_out, inputs, output) {
    auto sum_grad = gradient_registry[OpType::Sum](grad_out, inputs, output)[0];
    int64_t count = /* elements along reduced dim */;
    return {sum_grad / count};
};

// Max (returns gradient only to max element)
gradient_registry[OpType::Max] = [](grad_out, inputs, output) {
    auto x = inputs[0];
    auto max_val = output;
    // Create mask where x equals max
    auto mask = (x == broadcast_to(max_val, x->shape()));
    // Divide by count of max values (for ties)
    auto count = mask.sum();
    return {grad_out * mask / count};
};
```

---

## Appendix B: Python Module Structure

```
python/pyflame/
├── __init__.py           # Main exports
├── autograd/
│   ├── __init__.py
│   ├── grad_mode.py      # no_grad, enable_grad
│   └── function.py       # Autograd function base
├── nn/
│   ├── __init__.py       # Module, Sequential, etc.
│   ├── linear.py
│   ├── conv.py
│   ├── normalization.py
│   ├── pooling.py
│   ├── dropout.py
│   ├── attention.py
│   ├── embedding.py
│   ├── loss.py
│   └── functional.py     # Functional versions of layers
├── optim/
│   ├── __init__.py
│   ├── optimizer.py
│   ├── sgd.py
│   ├── adam.py
│   └── lr_scheduler.py
└── utils/
    ├── __init__.py
    ├── data.py           # DataLoader, Dataset
    └── checkpoint.py     # save, load
```

---

*Document Version: 1.0*
*Last Updated: January 10, 2026*
