# Lazy Evaluation and Graph Building System

**PyFlame Version:** Pre-Release Alpha 1.0
**Document Version:** 1.0
**Last Updated:** January 10, 2026
**Status:** Design Phase

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs and designs described here are subject to change.

---

## 1. Overview

PyFlame uses **lazy evaluation** as a fundamental design principle. Unlike eager execution frameworks (standard PyTorch), PyFlame defers computation until explicitly requested. This design is essential for the Cerebras WSE because:

1. **Compilation Cost**: Compiling a computation graph to CSL is expensive (seconds to minutes). We must batch operations.
2. **Whole-Program Optimization**: The WSE benefits from seeing the entire computation graph for optimal PE placement and routing.
3. **Static Graphs**: The WSE cannot be reprogrammed mid-execution. The entire graph must be known upfront.
4. **Memory Planning**: With only 48KB per PE, precise memory allocation requires full graph knowledge.

---

## 2. Design Philosophy

### 2.1 Lazy vs Eager: Trade-offs

| Aspect | Eager (PyTorch) | Lazy (PyFlame) |
|--------|-----------------|----------------|
| **Debugging** | Easier - immediate results | Harder - deferred execution |
| **Dynamic Control Flow** | Natural | Requires explicit handling |
| **Optimization** | Limited to operator-level | Whole-graph optimization |
| **Compilation Overhead** | Per-operator (CUDA kernels) | Amortized across graph |
| **WSE Compatibility** | Poor (graph changes) | Native (static compilation) |

### 2.2 When Computation Happens

Computation is triggered ("materialized") when the user:

1. Explicitly calls `.eval()` or `.execute()`
2. Accesses tensor data (`.numpy()`, `.data()`, indexing)
3. Prints tensor values
4. Converts to another framework's tensor

```python
import pyflame as pf

# No computation yet - just building the graph
a = pf.randn([1024, 1024])  # Creates graph node
b = pf.randn([1024, 1024])  # Creates graph node
c = a @ b                    # Creates MATMUL node
d = pf.relu(c)              # Creates RELU node
e = d.sum()                 # Creates REDUCE_SUM node

# NOW computation happens - graph is compiled and executed
result = e.eval()           # Returns scalar value

# OR - implicit materialization
print(e)                    # Forces evaluation
arr = e.numpy()             # Forces evaluation
```

---

## 3. Computation Graph Architecture

### 3.1 Graph Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Computation Graph                            │
├─────────────────────────────────────────────────────────────────────┤
│  Nodes:                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐            │
│  │ INPUT_A │   │ INPUT_B │   │ MATMUL  │   │  RELU   │            │
│  │ (Leaf)  │   │ (Leaf)  │   │         │   │         │            │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘            │
│       │             │             │             │                   │
│       └─────────────┴──────┬──────┴─────────────┘                   │
│                            │                                        │
│  Edges: Data dependencies (tensor flow)                            │
│  Metadata: Shapes, dtypes, layouts, optimization hints             │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Node Types

```cpp
// include/pyflame/ir/node.hpp
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <variant>
#include "pyflame/core/dtype.hpp"

namespace pyflame::ir {

// Unique identifier for graph nodes
using NodeId = uint64_t;

// Types of nodes in the computation graph
enum class NodeType {
    // Data nodes
    CONSTANT,           // Immutable data (weights, literals)
    PARAMETER,          // Trainable parameter
    INPUT,              // Runtime input from host
    INTERMEDIATE,       // Result of computation

    // Operation nodes (these produce INTERMEDIATE data nodes)
    OP_ELEMENTWISE,     // add, mul, relu, etc.
    OP_REDUCTION,       // sum, mean, max, etc.
    OP_MATMUL,          // Matrix multiplication variants
    OP_RESHAPE,         // View, reshape, squeeze, unsqueeze
    OP_TRANSPOSE,       // Dimension permutation
    OP_SLICE,           // Tensor slicing
    OP_CONCAT,          // Concatenation
    OP_BROADCAST,       // Implicit broadcasting
};

// Forward declarations
class Node;
class Graph;

// Represents the shape and type of a tensor
struct TensorSpec {
    std::vector<int64_t> shape;
    DType dtype;
    MeshLayout layout;

    int64_t numel() const;
    size_t size_bytes() const;
};

// A node in the computation graph
class Node {
public:
    NodeId id() const { return id_; }
    NodeType type() const { return type_; }
    const std::string& name() const { return name_; }

    // Tensor specification of output
    const TensorSpec& output_spec() const { return output_spec_; }

    // Input edges (dependencies)
    const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }

    // Users of this node's output
    const std::vector<std::weak_ptr<Node>>& users() const { return users_; }

    // For operation nodes: the specific operation
    OpType op_type() const { return op_type_; }

    // For constant nodes: the actual data
    template<typename T>
    const T* constant_data() const;

    // Check if this node has been evaluated
    bool is_evaluated() const { return evaluated_; }

    // Mark as evaluated with result location
    void mark_evaluated(void* result_ptr);

private:
    friend class Graph;

    NodeId id_;
    NodeType type_;
    std::string name_;
    TensorSpec output_spec_;
    std::vector<std::shared_ptr<Node>> inputs_;
    std::vector<std::weak_ptr<Node>> users_;
    OpType op_type_ = OpType::NONE;
    bool evaluated_ = false;
    void* result_ptr_ = nullptr;

    // For constants
    std::vector<uint8_t> constant_data_;
};

}  // namespace pyflame::ir
```

### 3.3 Graph Class

```cpp
// include/pyflame/ir/graph.hpp
#pragma once

#include <unordered_map>
#include <memory>
#include "pyflame/ir/node.hpp"

namespace pyflame::ir {

class Graph {
public:
    Graph();
    ~Graph();

    // Node creation
    std::shared_ptr<Node> create_constant(
        const void* data,
        const TensorSpec& spec,
        const std::string& name = ""
    );

    std::shared_ptr<Node> create_input(
        const TensorSpec& spec,
        const std::string& name = ""
    );

    std::shared_ptr<Node> create_parameter(
        const TensorSpec& spec,
        const std::string& name = ""
    );

    std::shared_ptr<Node> create_op(
        OpType op_type,
        const std::vector<std::shared_ptr<Node>>& inputs,
        const TensorSpec& output_spec,
        const std::string& name = ""
    );

    // Graph access
    std::shared_ptr<Node> node(NodeId id) const;
    const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }
    const std::vector<std::shared_ptr<Node>>& outputs() const { return outputs_; }
    const std::vector<std::shared_ptr<Node>>& all_nodes() const { return all_nodes_; }

    // Mark nodes as outputs (for evaluation)
    void mark_output(std::shared_ptr<Node> node);

    // Topological ordering
    std::vector<std::shared_ptr<Node>> topological_order() const;

    // Subgraph extraction (for partial evaluation)
    Graph subgraph(const std::vector<std::shared_ptr<Node>>& outputs) const;

    // Debug printing
    std::string to_string() const;
    void dump_graphviz(const std::string& filename) const;

    // Graph statistics
    size_t num_nodes() const { return all_nodes_.size(); }
    size_t num_ops() const;
    size_t estimated_memory_bytes() const;

private:
    NodeId next_id_ = 0;
    std::vector<std::shared_ptr<Node>> all_nodes_;
    std::vector<std::shared_ptr<Node>> inputs_;
    std::vector<std::shared_ptr<Node>> outputs_;
    std::unordered_map<NodeId, std::shared_ptr<Node>> node_map_;

    NodeId allocate_id();
    void add_edge(std::shared_ptr<Node> from, std::shared_ptr<Node> to);
};

}  // namespace pyflame::ir
```

---

## 4. Graph Building (Tracing)

### 4.1 Tensor as Graph Handle

The user-facing `Tensor` class is actually a thin handle to a graph node:

```cpp
// include/pyflame/core/tensor_impl.hpp
#pragma once

#include <memory>
#include "pyflame/ir/graph.hpp"
#include "pyflame/ir/node.hpp"

namespace pyflame {

// The actual implementation behind Tensor
class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    // Construction from graph node (lazy tensor)
    static std::shared_ptr<TensorImpl> from_node(
        std::shared_ptr<ir::Graph> graph,
        std::shared_ptr<ir::Node> node
    );

    // Construction from data (immediate tensor - creates constant node)
    static std::shared_ptr<TensorImpl> from_data(
        const void* data,
        const std::vector<int64_t>& shape,
        DType dtype
    );

    // Access the underlying graph node
    std::shared_ptr<ir::Node> node() const { return node_; }

    // Access the graph this tensor belongs to
    std::shared_ptr<ir::Graph> graph() const { return graph_; }

    // Metadata accessors
    const std::vector<int64_t>& shape() const { return node_->output_spec().shape; }
    DType dtype() const { return node_->output_spec().dtype; }
    const MeshLayout& layout() const { return node_->output_spec().layout; }

    // Check if tensor has been materialized
    bool is_materialized() const { return node_->is_evaluated(); }

    // Force materialization and return data pointer
    void* materialize();

    // Create new tensor from operation on this tensor
    std::shared_ptr<TensorImpl> apply_unary(OpType op) const;
    std::shared_ptr<TensorImpl> apply_binary(OpType op, std::shared_ptr<TensorImpl> other) const;

private:
    std::shared_ptr<ir::Graph> graph_;
    std::shared_ptr<ir::Node> node_;

    // Cached materialized data (nullptr until evaluated)
    mutable std::shared_ptr<void> materialized_data_;
};

}  // namespace pyflame
```

### 4.2 Operation Recording

When the user performs an operation, we record it in the graph:

```cpp
// src/core/tensor_ops.cpp
#include "pyflame/core/tensor.hpp"
#include "pyflame/core/tensor_impl.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/ir/shape_inference.hpp"

namespace pyflame {

Tensor Tensor::operator+(const Tensor& other) const {
    auto& graph = impl_->graph();

    // Infer output shape (handles broadcasting)
    auto output_spec = ir::infer_binary_op_spec(
        impl_->node()->output_spec(),
        other.impl_->node()->output_spec(),
        ir::OpType::ADD
    );

    // Create operation node
    auto op_node = graph->create_op(
        ir::OpType::ADD,
        {impl_->node(), other.impl_->node()},
        output_spec,
        "add_" + std::to_string(graph->num_nodes())
    );

    // Return lazy tensor referencing the new node
    return Tensor(TensorImpl::from_node(graph, op_node));
}

Tensor relu(const Tensor& x) {
    auto& graph = x.impl_->graph();

    // ReLU preserves shape and dtype
    auto output_spec = x.impl_->node()->output_spec();

    auto op_node = graph->create_op(
        ir::OpType::RELU,
        {x.impl_->node()},
        output_spec,
        "relu_" + std::to_string(graph->num_nodes())
    );

    return Tensor(TensorImpl::from_node(graph, op_node));
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    auto& graph = a.impl_->graph();

    // Validate and infer matmul output shape
    auto output_spec = ir::infer_matmul_spec(
        a.impl_->node()->output_spec(),
        b.impl_->node()->output_spec()
    );

    auto op_node = graph->create_op(
        ir::OpType::MATMUL,
        {a.impl_->node(), b.impl_->node()},
        output_spec,
        "matmul_" + std::to_string(graph->num_nodes())
    );

    return Tensor(TensorImpl::from_node(graph, op_node));
}

}  // namespace pyflame
```

### 4.3 Shape Inference

```cpp
// include/pyflame/ir/shape_inference.hpp
#pragma once

#include "pyflame/ir/node.hpp"

namespace pyflame::ir {

// Infer output spec for binary operations (with broadcasting)
TensorSpec infer_binary_op_spec(
    const TensorSpec& a,
    const TensorSpec& b,
    OpType op
);

// Infer output spec for reduction operations
TensorSpec infer_reduction_spec(
    const TensorSpec& input,
    int dim,
    bool keepdim
);

// Infer output spec for matrix multiplication
TensorSpec infer_matmul_spec(
    const TensorSpec& a,
    const TensorSpec& b
);

// Infer output spec for reshape
TensorSpec infer_reshape_spec(
    const TensorSpec& input,
    const std::vector<int64_t>& new_shape
);

// Compute broadcasted shape
std::vector<int64_t> broadcast_shapes(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
);

}  // namespace pyflame::ir
```

---

## 5. Graph Context Management

### 5.1 Thread-Local Graph Context

Multiple computations should be independent by default:

```cpp
// include/pyflame/core/context.hpp
#pragma once

#include <memory>
#include <stack>
#include "pyflame/ir/graph.hpp"

namespace pyflame {

// Manages the current computation context
class Context {
public:
    // Get the current context (creates one if needed)
    static Context& current();

    // Get/set the active graph
    std::shared_ptr<ir::Graph> graph() const;
    void set_graph(std::shared_ptr<ir::Graph> graph);

    // Create a new scope with fresh graph
    class Scope {
    public:
        Scope();
        ~Scope();

        std::shared_ptr<ir::Graph> graph() const { return graph_; }

    private:
        std::shared_ptr<ir::Graph> graph_;
        std::shared_ptr<ir::Graph> previous_;
    };

    // Create a scope that reuses existing graph
    class ReuseScope {
    public:
        explicit ReuseScope(std::shared_ptr<ir::Graph> graph);
        ~ReuseScope();
    private:
        std::shared_ptr<ir::Graph> previous_;
    };

private:
    std::shared_ptr<ir::Graph> current_graph_;

    // Thread-local storage
    static thread_local Context* instance_;
};

// RAII helper for scoped graph contexts
#define PYFLAME_GRAPH_SCOPE() ::pyflame::Context::Scope _pf_scope_##__LINE__

}  // namespace pyflame
```

### 5.2 Usage Patterns

```cpp
// Pattern 1: Automatic context (most common)
Tensor a = Tensor::randn({1024, 1024});  // Creates graph in current context
Tensor b = Tensor::randn({1024, 1024});  // Same graph
Tensor c = a @ b;                         // Same graph

// Pattern 2: Explicit scope for isolation
{
    Context::Scope scope;  // New graph
    Tensor x = Tensor::randn({512, 512});
    Tensor y = x.sum();
    y.eval();  // Compiles and runs just this graph
}  // Scope ends, graph can be garbage collected

// Pattern 3: Multiple independent graphs
auto graph1 = std::make_shared<ir::Graph>();
auto graph2 = std::make_shared<ir::Graph>();

{
    Context::ReuseScope use_graph1(graph1);
    Tensor a = Tensor::randn({100, 100});
    // Operations go to graph1
}

{
    Context::ReuseScope use_graph2(graph2);
    Tensor b = Tensor::randn({200, 200});
    // Operations go to graph2
}
```

---

## 6. Materialization (Evaluation)

### 6.1 Evaluation Pipeline

```
User calls: tensor.eval()
                │
                ▼
┌─────────────────────────────────────────────┐
│       Graph Extraction                       │
│  - Find all nodes needed for this output    │
│  - Create subgraph if partial eval          │
└─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│       Optimization Passes                    │
│  - Constant folding                         │
│  - Dead code elimination                    │
│  - Common subexpression elimination         │
│  - Operator fusion                          │
│  - Layout optimization                      │
└─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│       Code Generation                        │
│  - Generate CSL from optimized graph        │
│  - See: 01_csl_code_generation.md           │
└─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│       Compilation                            │
│  - Invoke cslc to compile CSL               │
│  - Cache compiled kernels                   │
└─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│       Execution                              │
│  - Load kernel to WSE (or simulator)        │
│  - Transfer input data                      │
│  - Execute                                  │
│  - Transfer output data                     │
└─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│       Result Storage                         │
│  - Cache result in tensor node              │
│  - Return to user                           │
└─────────────────────────────────────────────┘
```

### 6.2 Executor Class

```cpp
// include/pyflame/runtime/executor.hpp
#pragma once

#include <memory>
#include <unordered_map>
#include "pyflame/ir/graph.hpp"
#include "pyflame/backend/csl_codegen.hpp"

namespace pyflame::runtime {

// Result of graph execution
struct ExecutionResult {
    bool success;
    std::string error_message;
    double compile_time_ms;
    double transfer_time_ms;
    double compute_time_ms;

    // Output data (mapped by node ID)
    std::unordered_map<ir::NodeId, std::shared_ptr<void>> outputs;
};

// Execution configuration
struct ExecutorConfig {
    bool use_cache = true;              // Cache compiled kernels
    bool profile = false;               // Enable profiling
    std::string device = "simulator";   // "simulator", "cs2", "cs3"
    int timeout_seconds = 300;
};

class Executor {
public:
    explicit Executor(ExecutorConfig config = {});
    ~Executor();

    // Execute a graph, computing specified output nodes
    ExecutionResult execute(
        std::shared_ptr<ir::Graph> graph,
        const std::vector<ir::NodeId>& output_ids
    );

    // Execute and return a single output
    template<typename T>
    std::vector<T> execute_single(
        std::shared_ptr<ir::Graph> graph,
        ir::NodeId output_id
    );

    // Warm up: compile without executing
    bool compile_only(std::shared_ptr<ir::Graph> graph);

    // Clear kernel cache
    void clear_cache();

    // Get cache statistics
    struct CacheStats {
        size_t hits;
        size_t misses;
        size_t cached_kernels;
        size_t cache_size_bytes;
    };
    CacheStats cache_stats() const;

private:
    ExecutorConfig config_;
    std::unique_ptr<backend::CSLCodeGenerator> codegen_;

    // Kernel cache (hash of graph structure -> compiled binary)
    struct CachedKernel {
        std::filesystem::path elf_path;
        std::chrono::system_clock::time_point compile_time;
    };
    std::unordered_map<size_t, CachedKernel> kernel_cache_;

    // Compute cache key for a graph
    size_t compute_cache_key(const ir::Graph& graph) const;

    // Run optimization passes
    void optimize(ir::Graph& graph);

    // Execute on simulator
    ExecutionResult execute_simulator(
        const std::filesystem::path& elf,
        const ir::Graph& graph,
        const std::vector<ir::NodeId>& output_ids
    );

    // Execute on hardware
    ExecutionResult execute_hardware(
        const std::filesystem::path& elf,
        const ir::Graph& graph,
        const std::vector<ir::NodeId>& output_ids
    );
};

}  // namespace pyflame::runtime
```

### 6.3 Tensor Materialization Implementation

```cpp
// src/core/tensor_impl.cpp
#include "pyflame/core/tensor_impl.hpp"
#include "pyflame/runtime/executor.hpp"
#include "pyflame/core/context.hpp"

namespace pyflame {

void* TensorImpl::materialize() {
    // Already materialized?
    if (materialized_data_) {
        return materialized_data_.get();
    }

    // Get executor from context
    auto& executor = Context::current().executor();

    // Execute graph for this node
    auto result = executor.execute(graph_, {node_->id()});

    if (!result.success) {
        throw std::runtime_error("Execution failed: " + result.error_message);
    }

    // Store result
    materialized_data_ = result.outputs.at(node_->id());
    node_->mark_evaluated(materialized_data_.get());

    return materialized_data_.get();
}

}  // namespace pyflame
```

---

## 7. Optimization Passes

### 7.1 Pass Infrastructure

```cpp
// include/pyflame/ir/passes/pass.hpp
#pragma once

#include "pyflame/ir/graph.hpp"

namespace pyflame::ir::passes {

// Base class for optimization passes
class Pass {
public:
    virtual ~Pass() = default;

    // Human-readable name
    virtual std::string name() const = 0;

    // Run the pass on a graph, returns true if modified
    virtual bool run(Graph& graph) = 0;
};

// Run a sequence of passes
class PassManager {
public:
    void add_pass(std::unique_ptr<Pass> pass);

    // Run all passes until fixed point
    void run(Graph& graph, int max_iterations = 10);

    // Standard optimization pipeline
    static PassManager create_default();

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

}  // namespace pyflame::ir::passes
```

### 7.2 Key Optimization Passes

```cpp
// include/pyflame/ir/passes/standard_passes.hpp
#pragma once

#include "pyflame/ir/passes/pass.hpp"

namespace pyflame::ir::passes {

// Fold constant operations at compile time
class ConstantFolding : public Pass {
public:
    std::string name() const override { return "ConstantFolding"; }
    bool run(Graph& graph) override;

private:
    bool can_fold(const Node& node) const;
    std::shared_ptr<Node> fold(Graph& graph, const Node& node);
};

// Remove operations whose results are unused
class DeadCodeElimination : public Pass {
public:
    std::string name() const override { return "DeadCodeElimination"; }
    bool run(Graph& graph) override;

private:
    std::set<NodeId> find_live_nodes(const Graph& graph) const;
};

// Merge identical computations
class CommonSubexpressionElimination : public Pass {
public:
    std::string name() const override { return "CSE"; }
    bool run(Graph& graph) override;

private:
    size_t hash_node(const Node& node) const;
    bool nodes_equivalent(const Node& a, const Node& b) const;
};

// Fuse compatible adjacent operations
class OperatorFusion : public Pass {
public:
    std::string name() const override { return "OperatorFusion"; }
    bool run(Graph& graph) override;

private:
    bool can_fuse(const Node& producer, const Node& consumer) const;
    std::shared_ptr<Node> fuse(Graph& graph, const Node& producer, const Node& consumer);
};

// Optimize tensor layouts for target architecture
class LayoutOptimization : public Pass {
public:
    std::string name() const override { return "LayoutOptimization"; }
    bool run(Graph& graph) override;

private:
    MeshLayout optimal_layout(const Node& node) const;
    void insert_layout_conversion(Graph& graph, std::shared_ptr<Node> node);
};

// Algebraic simplifications (a + 0 = a, a * 1 = a, etc.)
class AlgebraicSimplification : public Pass {
public:
    std::string name() const override { return "AlgebraicSimplification"; }
    bool run(Graph& graph) override;
};

}  // namespace pyflame::ir::passes
```

### 7.3 Operator Fusion Details

Fusion is critical for WSE performance. We define fusion patterns:

```cpp
// src/ir/passes/operator_fusion.cpp

// Fuseable patterns for elementwise chains:
//   relu(add(a, b))    -> fused_add_relu(a, b)
//   sigmoid(mul(a, b)) -> fused_mul_sigmoid(a, b)
//   add(mul(a, b), c)  -> fused_fma(a, b, c)  // Fused multiply-add

// Fuseable patterns for matmul + activation:
//   relu(matmul(a, b))    -> fused_matmul_relu(a, b)
//   gelu(matmul(a, b))    -> fused_matmul_gelu(a, b)

// Fuseable patterns for reduction:
//   mean(x) is rewritten as: div(sum(x), numel(x))

bool OperatorFusion::can_fuse(const Node& producer, const Node& consumer) const {
    // Only fuse if consumer has single input from producer
    if (consumer.inputs().size() != 1) return false;
    if (producer.users().size() != 1) return false;

    // Check fusion compatibility
    OpType prod_op = producer.op_type();
    OpType cons_op = consumer.op_type();

    // Elementwise -> Elementwise: always fuseable
    if (is_elementwise(prod_op) && is_elementwise(cons_op)) {
        return true;
    }

    // Matmul -> Activation: fuseable
    if (prod_op == OpType::MATMUL && is_activation(cons_op)) {
        return true;
    }

    return false;
}
```

---

## 8. Graph Serialization

### 8.1 Serialization Format

For caching compiled graphs and debugging:

```cpp
// include/pyflame/ir/serialization.hpp
#pragma once

#include <iostream>
#include "pyflame/ir/graph.hpp"

namespace pyflame::ir {

// Serialize graph to binary format
void serialize_graph(const Graph& graph, std::ostream& out);

// Deserialize graph from binary format
Graph deserialize_graph(std::istream& in);

// Human-readable text format (for debugging)
std::string graph_to_text(const Graph& graph);
Graph text_to_graph(const std::string& text);

// GraphViz DOT format (for visualization)
std::string graph_to_dot(const Graph& graph);

}  // namespace pyflame::ir
```

### 8.2 Binary Format Specification

```
PyFlame Graph Binary Format (PFGB)
==================================

Header (16 bytes):
  - Magic: "PFGB" (4 bytes)
  - Version: uint32_t (4 bytes)
  - Flags: uint32_t (4 bytes)
  - Node count: uint32_t (4 bytes)

For each node:
  - Node ID: uint64_t
  - Node type: uint8_t
  - Op type: uint8_t (if operation node)
  - Name length: uint16_t
  - Name: utf8 string
  - Output spec:
    - Shape rank: uint8_t
    - Shape dims: int64_t * rank
    - DType: uint8_t
    - Layout type: uint8_t
    - Layout params: variable
  - Input count: uint16_t
  - Input IDs: uint64_t * input_count
  - Constant data (if constant node):
    - Data length: uint64_t
    - Data: bytes

Footer:
  - Output node count: uint32_t
  - Output node IDs: uint64_t * count
  - Checksum: uint32_t (CRC32)
```

---

## 9. Python Integration

### 9.1 Python API for Lazy Tensors

```python
# python/pyflame/lazy.py
"""Lazy evaluation control for PyFlame tensors."""

import pyflame._pyflame_cpp as _cpp

def is_lazy(tensor):
    """Check if tensor has not been evaluated yet."""
    return not tensor._impl.is_materialized()

def eval(*tensors):
    """Force evaluation of one or more tensors.

    Args:
        *tensors: Variable number of tensors to evaluate.

    Returns:
        The evaluated tensors (same objects, but now materialized).

    Example:
        >>> a = pf.randn([1000, 1000])
        >>> b = pf.randn([1000, 1000])
        >>> c = a @ b
        >>> pf.is_lazy(c)  # True
        >>> pf.eval(c)
        >>> pf.is_lazy(c)  # False
    """
    for t in tensors:
        t._impl.materialize()
    return tensors[0] if len(tensors) == 1 else tensors

def compile_graph(output_tensor, optimize=True):
    """Compile the graph leading to a tensor without executing.

    Useful for checking compilation time and debugging.

    Args:
        output_tensor: The tensor whose graph should be compiled.
        optimize: Whether to run optimization passes.

    Returns:
        CompilationInfo with timing and statistics.
    """
    return _cpp.compile_graph(output_tensor._impl, optimize)

def get_graph_stats(tensor):
    """Get statistics about the computation graph.

    Returns:
        dict with keys: 'num_nodes', 'num_ops', 'estimated_memory_bytes'
    """
    graph = tensor._impl.graph()
    return {
        'num_nodes': graph.num_nodes(),
        'num_ops': graph.num_ops(),
        'estimated_memory_bytes': graph.estimated_memory_bytes(),
    }

def print_graph(tensor, file=None):
    """Print the computation graph for debugging.

    Args:
        tensor: Tensor whose graph to print.
        file: File to write to (default: stdout).
    """
    graph = tensor._impl.graph()
    text = graph.to_string()
    print(text, file=file)
```

### 9.2 Context Managers for Scope Control

```python
# python/pyflame/context.py
"""Context management for PyFlame computation graphs."""

import pyflame._pyflame_cpp as _cpp
from contextlib import contextmanager

@contextmanager
def graph_scope():
    """Create an isolated graph scope.

    Operations within this scope go to a new, independent graph.

    Example:
        >>> with pf.graph_scope():
        ...     a = pf.randn([100])
        ...     b = a.sum()
        ...     result = pf.eval(b)
        >>> # Graph is released after scope
    """
    scope = _cpp.Context.Scope()
    try:
        yield scope.graph()
    finally:
        del scope

@contextmanager
def reuse_graph(graph):
    """Reuse an existing graph for operations.

    Args:
        graph: Graph object to use.

    Example:
        >>> g = pf.create_graph()
        >>> with pf.reuse_graph(g):
        ...     a = pf.randn([100])
        >>> with pf.reuse_graph(g):
        ...     b = pf.randn([200])  # Same graph as 'a'
    """
    scope = _cpp.Context.ReuseScope(graph)
    try:
        yield
    finally:
        del scope

def create_graph():
    """Create a new, empty computation graph."""
    return _cpp.Graph()

def current_graph():
    """Get the current context's computation graph."""
    return _cpp.Context.current().graph()
```

---

## 10. Example: Complete Lazy Evaluation Flow

```python
import pyflame as pf

# Phase 1: Graph Building
# -----------------------
# No computation happens yet

print("Building graph...")

# These create nodes in the graph
x = pf.randn([1024, 512])       # Node 0: INPUT
w1 = pf.randn([512, 256])       # Node 1: INPUT
b1 = pf.zeros([256])            # Node 2: CONSTANT

# These create operation nodes
h = x @ w1                       # Node 3: MATMUL(0, 1)
h = h + b1                       # Node 4: ADD(3, 2)
h = pf.relu(h)                  # Node 5: RELU(4)

w2 = pf.randn([256, 10])        # Node 6: INPUT
out = h @ w2                     # Node 7: MATMUL(5, 6)
out = pf.softmax(out, dim=1)   # Node 8: SOFTMAX(7)

loss = -out.log().sum()         # Nodes 9, 10: LOG(8), REDUCE_SUM(9), NEG(10)

# Check: nothing computed yet
print(f"Graph has {pf.get_graph_stats(loss)['num_nodes']} nodes")
print(f"loss is lazy: {pf.is_lazy(loss)}")

# Phase 2: Optimization
# ---------------------
# Happens automatically at eval() time, but can be inspected

pf.print_graph(loss)  # See the raw graph

# Phase 3: Evaluation
# -------------------
# NOW computation happens

print("\nEvaluating...")
result = pf.eval(loss)  # Compile + Execute

print(f"Loss value: {result.numpy()}")
print(f"loss is lazy: {pf.is_lazy(loss)}")  # False now
```

Output:
```
Building graph...
Graph has 11 nodes
loss is lazy: True

Graph:
  %0 = input([1024, 512], f32)
  %1 = input([512, 256], f32)
  %2 = constant([256], f32)
  %3 = matmul(%0, %1) -> [1024, 256]
  %4 = add(%3, %2) -> [1024, 256]
  %5 = relu(%4) -> [1024, 256]
  %6 = input([256, 10], f32)
  %7 = matmul(%5, %6) -> [1024, 10]
  %8 = softmax(%7, dim=1) -> [1024, 10]
  %9 = log(%8) -> [1024, 10]
  %10 = reduce_sum(%9) -> []
  %11 = neg(%10) -> []
  outputs: [%11]

Evaluating...
Loss value: 23041.5
loss is lazy: False
```

---

## 11. Future Work

### 11.1 Planned Features

1. **Just-In-Time Compilation**: Background compilation while user code runs
2. **Graph Caching**: Persistent cache of compiled graphs across sessions
3. **Partial Evaluation**: Evaluate subgraphs while keeping others lazy
4. **Dynamic Shapes**: Support for graphs with symbolic dimensions
5. **Control Flow**: Support for conditionals and loops in graphs

### 11.2 Research Areas

1. **Automatic Differentiation**: Extend graph with gradient computation
2. **Distributed Graphs**: Graphs spanning multiple WSE chips
3. **Mixed Precision**: Automatic precision selection per operation

---

*Document Version: 1.0*
*Authors: PyFlame Team*
*Last Updated: January 10, 2026*
