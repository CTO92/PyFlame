# CSL Code Generation Strategy

**PyFlame Version:** Pre-Release Alpha 1.0
**Document Version:** 1.0
**Last Updated:** January 10, 2026
**Status:** Design Phase

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs and designs described here are subject to change.

---

## 1. Overview

The CSL (Cerebras Software Language) code generation subsystem is responsible for transforming PyFlame's high-level tensor operations into executable CSL kernels that run on the Cerebras WSE. This document describes the architecture, strategies, and implementation details for this critical component.

### 1.1 Design Goals

1. **Correctness First**: Generated CSL must be semantically equivalent to the user's intent
2. **Performance**: Maximize PE utilization, minimize wavelet communication overhead
3. **Maintainability**: Template-based approach with clear separation of concerns
4. **Extensibility**: Easy to add new operations without rewriting core generation logic
5. **Debuggability**: Generated code should be readable and traceable to source operations

---

## 2. Architecture

### 2.1 Code Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PyFlame Operation Graph                          │
│  (After optimization passes: fusion, layout selection, etc.)        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Operation Lowering                               │
│  - Map high-level ops to CSL primitives                            │
│  - Determine PE assignment for each operation                       │
│  - Compute data dependencies and wavelet routing                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CSL Template Selection                           │
│  - Select appropriate kernel template per op type                  │
│  - Parameterize templates with tile sizes, dtypes, etc.            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Routing Table Generation                         │
│  - Compute color assignments for wavelet routes                    │
│  - Generate fabric description (layout.csl)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CSL Source Emission                              │
│  - Generate pe_program.csl for each PE role                        │
│  - Generate layout.csl with fabric topology                        │
│  - Generate run.py launcher script                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CSL Compiler Invocation                          │
│  - Call `cslc` to compile to WSE binary                            │
│  - Handle compilation errors/warnings                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

```
include/pyflame/backend/
├── csl_codegen.hpp          # Main code generator interface
├── csl_emitter.hpp          # Low-level CSL text emission
├── csl_templates.hpp        # Template registry and instantiation
├── routing_planner.hpp      # Wavelet route computation
├── pe_assignment.hpp        # Operation → PE mapping
└── csl_compiler.hpp         # CSL compiler invocation wrapper

src/backend/
├── csl_codegen.cpp
├── csl_emitter.cpp
├── csl_templates/
│   ├── elementwise.csl.template
│   ├── reduction.csl.template
│   ├── matmul_single_pe.csl.template
│   ├── matmul_tiled.csl.template
│   └── layout.csl.template
├── routing_planner.cpp
├── pe_assignment.cpp
└── csl_compiler.cpp
```

---

## 3. Operation Lowering

### 3.1 Operation Classification

Operations are classified by their CSL implementation pattern:

| Category | Examples | CSL Pattern |
|----------|----------|-------------|
| **Elementwise** | add, mul, relu, sigmoid | Single task, no communication |
| **Reduction** | sum, mean, max | Tree-based wavelet aggregation |
| **Matmul** | matmul, bmm | Systolic array or tiled algorithm |
| **Data Movement** | transpose, reshape | Layout change, may require communication |
| **Communication** | gather, scatter | Explicit wavelet routing |

### 3.2 Lowering Interface

```cpp
// include/pyflame/backend/lowering.hpp
#pragma once

#include "pyflame/ir/operation.hpp"
#include "pyflame/backend/csl_primitive.hpp"

namespace pyflame::backend {

// A lowered operation ready for CSL generation
struct LoweredOp {
    CSLPrimitiveType primitive;     // Which CSL pattern to use
    std::vector<PECoord> pe_set;    // Which PEs execute this
    TileSpec tile_spec;             // How data is tiled
    std::vector<WaveletRoute> routes; // Communication requirements
    std::map<std::string, std::string> params; // Template parameters
};

// Lowers a high-level op to CSL primitives
class OperationLowering {
public:
    LoweredOp lower(const ir::Operation& op, const MeshLayout& layout);

private:
    LoweredOp lower_elementwise(const ir::Operation& op, const MeshLayout& layout);
    LoweredOp lower_reduction(const ir::Operation& op, const MeshLayout& layout);
    LoweredOp lower_matmul(const ir::Operation& op, const MeshLayout& layout);
    LoweredOp lower_transpose(const ir::Operation& op, const MeshLayout& layout);
};

}  // namespace pyflame::backend
```

---

## 4. Template-Based Code Generation

### 4.1 Why Templates?

We use a **template-based approach** rather than pure programmatic generation for several reasons:

1. **Readability**: Templates look like actual CSL code, easier to verify
2. **Maintainability**: CSL experts can modify templates without C++ knowledge
3. **Debugging**: Generated code is recognizable and traceable
4. **Correctness**: Less string manipulation means fewer generation bugs

### 4.2 Template Format

Templates use a simple substitution syntax:

```csl
// templates/elementwise_binary.csl.template
// PyFlame Generated - DO NOT EDIT
// Operation: {{OP_NAME}}
// Generated: {{TIMESTAMP}}

param TILE_SIZE: u32 = {{TILE_SIZE}};

const input_a: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
const input_b: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
var output: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});

task compute() void {
    for (@range(u32, 0, TILE_SIZE)) |i| {
        output[i] = {{OPERATION_EXPR}};
    }
}

comptime {
    @set_local_task_id(compute, 0);
}
```

### 4.3 Template Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{{OP_NAME}}` | Human-readable operation name | `"add"` |
| `{{DTYPE}}` | CSL data type | `f32`, `f16`, `i32` |
| `{{TILE_SIZE}}` | Elements per PE | `1024` |
| `{{OPERATION_EXPR}}` | The actual operation expression | `input_a[i] + input_b[i]` |
| `{{PE_COORD_X}}` | PE X coordinate | `0` |
| `{{PE_COORD_Y}}` | PE Y coordinate | `0` |
| `{{NUM_WAVELETS}}` | Number of wavelets to send/receive | `32` |
| `{{COLOR}}` | Wavelet color for routing | `0` |

### 4.4 Template Registry

```cpp
// include/pyflame/backend/csl_templates.hpp
#pragma once

#include <string>
#include <map>
#include <filesystem>

namespace pyflame::backend {

enum class CSLTemplate {
    ELEMENTWISE_UNARY,
    ELEMENTWISE_BINARY,
    REDUCTION_TREE,
    REDUCTION_LOCAL,
    MATMUL_SINGLE_PE,
    MATMUL_SYSTOLIC_CELL,
    MATMUL_SYSTOLIC_CONTROLLER,
    TRANSPOSE_LOCAL,
    TRANSPOSE_COMMUNICATION,
    LAYOUT_FABRIC,
};

class TemplateRegistry {
public:
    static TemplateRegistry& instance();

    // Load templates from directory (called at startup)
    void load_templates(const std::filesystem::path& template_dir);

    // Get raw template content
    std::string get_template(CSLTemplate type) const;

    // Instantiate template with parameters
    std::string instantiate(
        CSLTemplate type,
        const std::map<std::string, std::string>& params
    ) const;

private:
    std::map<CSLTemplate, std::string> templates_;

    std::string substitute_params(
        const std::string& tmpl,
        const std::map<std::string, std::string>& params
    ) const;
};

}  // namespace pyflame::backend
```

### 4.5 Template Instantiation Implementation

```cpp
// src/backend/csl_templates.cpp
#include "pyflame/backend/csl_templates.hpp"
#include <regex>
#include <fstream>
#include <sstream>

namespace pyflame::backend {

std::string TemplateRegistry::substitute_params(
    const std::string& tmpl,
    const std::map<std::string, std::string>& params
) const {
    std::string result = tmpl;

    // Match {{PARAM_NAME}} patterns
    std::regex pattern(R"(\{\{(\w+)\}\})");

    std::string::const_iterator search_start(result.cbegin());
    std::smatch match;
    std::string output;

    while (std::regex_search(search_start, result.cend(), match, pattern)) {
        output += match.prefix();

        std::string param_name = match[1].str();
        auto it = params.find(param_name);
        if (it != params.end()) {
            output += it->second;
        } else {
            // Keep original if not found (for debugging)
            output += match[0].str();
        }

        search_start = match.suffix().first;
    }
    output += std::string(search_start, result.cend());

    return output;
}

std::string TemplateRegistry::instantiate(
    CSLTemplate type,
    const std::map<std::string, std::string>& params
) const {
    std::string tmpl = get_template(type);
    return substitute_params(tmpl, params);
}

}  // namespace pyflame::backend
```

---

## 5. Kernel Generation Examples

### 5.1 Elementwise Addition

**Input (PyFlame IR):**
```
%2 = add(%0, %1) : tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
     layout: SinglePE
```

**Generated CSL (pe_program.csl):**
```csl
// PyFlame Generated - Elementwise Add
// Source: Operation #2

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = 1,
    .height = 1,
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);

const TILE_SIZE: u32 = 1024;

var input_a: [TILE_SIZE]f32 = @zeros([TILE_SIZE]f32);
var input_b: [TILE_SIZE]f32 = @zeros([TILE_SIZE]f32);
var output: [TILE_SIZE]f32 = @zeros([TILE_SIZE]f32);

// Export symbols for host access
export var input_a_ptr = &input_a;
export var input_b_ptr = &input_b;
export var output_ptr = &output;

task compute() void {
    for (@range(u32, 0, TILE_SIZE)) |i| {
        output[i] = input_a[i] + input_b[i];
    }
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task_to_color(compute, sys_mod.LAUNCH);
}
```

### 5.2 ReLU Activation

**Input (PyFlame IR):**
```
%1 = relu(%0) : tensor<2048xf32> -> tensor<2048xf32>
     layout: RowPartition(2)  // Split across 2 PEs
```

**Generated CSL (pe_program.csl):**
```csl
// PyFlame Generated - ReLU
// PE Role: Compute PE (one of 2)

param pe_id: u32;  // 0 or 1

const TILE_SIZE: u32 = 1024;  // 2048 / 2 PEs

var input: [TILE_SIZE]f32 = @zeros([TILE_SIZE]f32);
var output: [TILE_SIZE]f32 = @zeros([TILE_SIZE]f32);

task compute() void {
    for (@range(u32, 0, TILE_SIZE)) |i| {
        const val = input[i];
        output[i] = if (val > 0.0) val else 0.0;
    }
}

comptime {
    @set_local_task_id(compute, 0);
}
```

### 5.3 Reduction (Sum)

**Input (PyFlame IR):**
```
%1 = reduce_sum(%0, dim=0) : tensor<1024x1024xf32> -> tensor<1024xf32>
     layout: Grid(4, 4)
```

**Generated CSL (reduction pattern with tree aggregation):**
```csl
// PyFlame Generated - Reduction Sum
// PE Role: Compute + Aggregate

param pe_x: u32;
param pe_y: u32;
param is_aggregator: bool;

const TILE_ROWS: u32 = 256;   // 1024 / 4
const TILE_COLS: u32 = 256;   // 1024 / 4

var local_data: [TILE_ROWS][TILE_COLS]f32;
var partial_sum: [TILE_COLS]f32 = @zeros([TILE_COLS]f32);
var received_sum: [TILE_COLS]f32 = @zeros([TILE_COLS]f32);

// Colors for reduction tree communication
const color_reduce_east: color = @get_color(0);
const color_reduce_south: color = @get_color(1);

// Fabric connections for reduction tree
const fabric_east = @get_output_queue(color_reduce_east);
const fabric_south = @get_input_queue(color_reduce_south);

task compute_local_sum() void {
    // Compute partial sum along dimension 0 (rows)
    for (@range(u32, 0, TILE_COLS)) |col| {
        var sum: f32 = 0.0;
        for (@range(u32, 0, TILE_ROWS)) |row| {
            sum += local_data[row][col];
        }
        partial_sum[col] = sum;
    }

    // If not leftmost PE, send to west neighbor
    if (pe_x > 0) {
        @activate(send_partial);
    } else {
        // Leftmost PEs aggregate
        @activate(aggregate);
    }
}

task send_partial() void {
    // Send partial sum via wavelet
    for (@range(u32, 0, TILE_COLS)) |i| {
        fabric_east.enqueue(partial_sum[i]);
    }
}

task aggregate() void {
    // Receive and accumulate from east neighbor(s)
    // (Simplified - actual implementation handles tree structure)
    for (@range(u32, 0, TILE_COLS)) |i| {
        const recv_val = fabric_south.dequeue();
        partial_sum[i] += recv_val;
    }
}

comptime {
    @set_local_task_id(compute_local_sum, 0);
    @bind_task_to_color(send_partial, color_reduce_east);
    @bind_task_to_color(aggregate, color_reduce_south);
}
```

---

## 6. Wavelet Routing Generation

### 6.1 Routing Planner

The routing planner computes color assignments and fabric routes for inter-PE communication.

```cpp
// include/pyflame/backend/routing_planner.hpp
#pragma once

#include <vector>
#include <map>
#include "pyflame/core/tensor.hpp"

namespace pyflame::backend {

// Direction for wavelet travel
enum class Direction { NORTH, SOUTH, EAST, WEST, RAMP };

// A single hop in a wavelet route
struct RouteHop {
    PECoord from;
    PECoord to;
    Direction dir;
    int color;
};

// Complete route from source PE to destination PE
struct WaveletRoute {
    PECoord source;
    PECoord dest;
    int color;
    std::vector<RouteHop> hops;
};

// Routing plan for entire computation graph
struct RoutingPlan {
    std::vector<WaveletRoute> routes;
    std::map<int, std::string> color_names;  // color_id -> symbolic name
    int max_colors_used;
};

class RoutingPlanner {
public:
    // Compute minimal routing for a set of communication requirements
    RoutingPlan plan_routes(
        const std::vector<std::pair<PECoord, PECoord>>& comm_pairs,
        const MeshLayout& layout
    );

private:
    // Use dimension-order routing (X then Y) to avoid deadlock
    std::vector<RouteHop> dimension_order_route(
        PECoord from, PECoord to, int color
    );

    // Assign colors to avoid conflicts on shared links
    std::map<int, int> assign_colors(
        const std::vector<std::vector<RouteHop>>& all_routes
    );
};

}  // namespace pyflame::backend
```

### 6.2 Layout Generation

The fabric layout describes PE connections and is compiled by `cslc`.

```csl
// Generated layout.csl for a 4x4 matmul grid

param memcpy_params: comptime_struct;

const GRID_ROWS: u32 = 4;
const GRID_COLS: u32 = 4;

// Color definitions for inter-PE communication
const color_a_broadcast: color = @get_color(0);  // Broadcast A tiles east
const color_b_broadcast: color = @get_color(1);  // Broadcast B tiles south
const color_c_accumulate: color = @get_color(2); // Accumulate C tiles

const memcpy = @import_module("<memcpy_multi/memcpy>", memcpy_params);

layout {
    @set_rectangle(GRID_COLS, GRID_ROWS);

    // Configure each PE
    for (@range(u32, 0, GRID_ROWS)) |row| {
        for (@range(u32, 0, GRID_COLS)) |col| {
            @set_tile_code(col, row, "pe_program.csl", .{
                .pe_x = col,
                .pe_y = row,
                .memcpy_params = memcpy.get_params(col),
            });
        }
    }

    // Configure horizontal (east) routes for A broadcast
    for (@range(u32, 0, GRID_ROWS)) |row| {
        for (@range(u32, 0, GRID_COLS - 1)) |col| {
            @set_color_config(col, row, color_a_broadcast, .{
                .routes = .{ .rx = .RAMP, .tx = .EAST }
            });
        }
    }

    // Configure vertical (south) routes for B broadcast
    for (@range(u32, 0, GRID_ROWS - 1)) |row| {
        for (@range(u32, 0, GRID_COLS)) |col| {
            @set_color_config(col, row, color_b_broadcast, .{
                .routes = .{ .rx = .RAMP, .tx = .SOUTH }
            });
        }
    }

    // Export memcpy infrastructure
    @export_name("input_a_ptr", [*]f32, true);
    @export_name("input_b_ptr", [*]f32, true);
    @export_name("output_ptr", [*]f32, false);
}
```

---

## 7. CSL Compiler Integration

### 7.1 Compiler Wrapper

```cpp
// include/pyflame/backend/csl_compiler.hpp
#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace pyflame::backend {

struct CompilationResult {
    bool success;
    std::filesystem::path output_elf;  // Compiled binary
    std::string stdout_log;
    std::string stderr_log;
    double compile_time_seconds;
};

struct CompilerOptions {
    std::filesystem::path sdk_path;         // Path to Cerebras SDK
    std::string target = "wse2";            // wse2 or wse3
    bool optimize = true;                   // Enable optimizations
    int fabric_width = 0;                   // Auto if 0
    int fabric_height = 0;
    bool generate_debug_info = false;
    std::vector<std::string> extra_flags;
};

class CSLCompiler {
public:
    explicit CSLCompiler(CompilerOptions options);

    // Compile a generated CSL project
    CompilationResult compile(
        const std::filesystem::path& layout_csl,
        const std::filesystem::path& output_dir
    );

    // Check if SDK is available
    bool is_sdk_available() const;

    // Get SDK version string
    std::string sdk_version() const;

private:
    CompilerOptions options_;
    std::filesystem::path cslc_path_;

    std::string build_command_line(
        const std::filesystem::path& layout_csl,
        const std::filesystem::path& output_dir
    ) const;
};

}  // namespace pyflame::backend
```

### 7.2 Compiler Invocation

```cpp
// src/backend/csl_compiler.cpp
#include "pyflame/backend/csl_compiler.hpp"
#include <cstdlib>
#include <array>
#include <chrono>

namespace pyflame::backend {

CompilationResult CSLCompiler::compile(
    const std::filesystem::path& layout_csl,
    const std::filesystem::path& output_dir
) {
    CompilationResult result;
    auto start = std::chrono::high_resolution_clock::now();

    // Build command: cslc layout.csl --fabric-dims=WxH --output-dir=...
    std::string cmd = build_command_line(layout_csl, output_dir);

    // Execute compiler (simplified - real impl uses proper subprocess handling)
    std::array<char, 4096> buffer;
    std::string stdout_capture;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        result.success = false;
        result.stderr_log = "Failed to launch cslc";
        return result;
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        stdout_capture += buffer.data();
    }

    int exit_code = pclose(pipe);

    auto end = std::chrono::high_resolution_clock::now();
    result.compile_time_seconds =
        std::chrono::duration<double>(end - start).count();

    result.success = (exit_code == 0);
    result.stdout_log = stdout_capture;
    result.output_elf = output_dir / "out.elf";

    return result;
}

std::string CSLCompiler::build_command_line(
    const std::filesystem::path& layout_csl,
    const std::filesystem::path& output_dir
) const {
    std::ostringstream cmd;
    cmd << cslc_path_.string();
    cmd << " " << layout_csl.string();

    if (options_.fabric_width > 0 && options_.fabric_height > 0) {
        cmd << " --fabric-dims=" << options_.fabric_width
            << "," << options_.fabric_height;
    }

    cmd << " --fabric-offsets=4,1";  // Standard offset for memcpy
    cmd << " -o " << output_dir.string();

    if (options_.optimize) {
        cmd << " -O3";
    }

    if (options_.generate_debug_info) {
        cmd << " -g";
    }

    for (const auto& flag : options_.extra_flags) {
        cmd << " " << flag;
    }

    return cmd.str();
}

}  // namespace pyflame::backend
```

---

## 8. Code Generator Implementation

### 8.1 Main Generator Class

```cpp
// include/pyflame/backend/csl_codegen.hpp
#pragma once

#include <filesystem>
#include "pyflame/ir/graph.hpp"
#include "pyflame/backend/csl_templates.hpp"
#include "pyflame/backend/routing_planner.hpp"
#include "pyflame/backend/csl_compiler.hpp"

namespace pyflame::backend {

struct CodeGenResult {
    std::filesystem::path output_dir;
    std::vector<std::filesystem::path> generated_files;
    std::optional<CompilationResult> compilation;
};

class CSLCodeGenerator {
public:
    CSLCodeGenerator();

    // Generate CSL from optimized graph
    CodeGenResult generate(
        const ir::Graph& graph,
        const std::filesystem::path& output_dir,
        bool compile = true
    );

    // Generate and return CSL source as strings (for debugging)
    std::map<std::string, std::string> generate_source(
        const ir::Graph& graph
    );

private:
    TemplateRegistry& templates_;
    RoutingPlanner router_;
    std::optional<CSLCompiler> compiler_;

    // Generate code for different operation types
    std::string gen_elementwise(const ir::Operation& op);
    std::string gen_reduction(const ir::Operation& op);
    std::string gen_matmul(const ir::Operation& op);
    std::string gen_layout(const ir::Graph& graph, const RoutingPlan& routes);

    // Write files to disk
    void write_files(
        const std::map<std::string, std::string>& sources,
        const std::filesystem::path& output_dir
    );
};

}  // namespace pyflame::backend
```

### 8.2 Generator Implementation

```cpp
// src/backend/csl_codegen.cpp
#include "pyflame/backend/csl_codegen.hpp"
#include <fstream>

namespace pyflame::backend {

CSLCodeGenerator::CSLCodeGenerator()
    : templates_(TemplateRegistry::instance())
{
    // Try to initialize compiler if SDK is available
    CompilerOptions opts;
    opts.sdk_path = std::getenv("CEREBRAS_SDK_PATH")
        ? std::getenv("CEREBRAS_SDK_PATH")
        : "/opt/cerebras/sdk";

    CSLCompiler compiler(opts);
    if (compiler.is_sdk_available()) {
        compiler_ = std::move(compiler);
    }
}

CodeGenResult CSLCodeGenerator::generate(
    const ir::Graph& graph,
    const std::filesystem::path& output_dir,
    bool compile
) {
    CodeGenResult result;
    result.output_dir = output_dir;

    // 1. Generate sources
    auto sources = generate_source(graph);

    // 2. Write to disk
    std::filesystem::create_directories(output_dir);
    write_files(sources, output_dir);

    for (const auto& [name, _] : sources) {
        result.generated_files.push_back(output_dir / name);
    }

    // 3. Compile if requested and SDK available
    if (compile && compiler_) {
        result.compilation = compiler_->compile(
            output_dir / "layout.csl",
            output_dir
        );
    }

    return result;
}

std::map<std::string, std::string> CSLCodeGenerator::generate_source(
    const ir::Graph& graph
) {
    std::map<std::string, std::string> sources;

    // Analyze graph for routing requirements
    std::vector<std::pair<PECoord, PECoord>> comm_pairs;
    for (const auto& op : graph.operations()) {
        // Extract communication requirements from data dependencies
        // ... (implementation detail)
    }

    RoutingPlan routes = router_.plan_routes(comm_pairs, graph.layout());

    // Generate PE programs
    for (const auto& op : graph.operations()) {
        std::string code;
        switch (op.type()) {
            case ir::OpType::ADD:
            case ir::OpType::MUL:
            case ir::OpType::RELU:
            case ir::OpType::SIGMOID:
                code = gen_elementwise(op);
                break;
            case ir::OpType::SUM:
            case ir::OpType::MEAN:
            case ir::OpType::MAX:
                code = gen_reduction(op);
                break;
            case ir::OpType::MATMUL:
                code = gen_matmul(op);
                break;
            default:
                throw std::runtime_error("Unsupported op for CSL: " + op.name());
        }
        sources["pe_program.csl"] = code;  // Simplified - real impl handles multiple PEs
    }

    // Generate fabric layout
    sources["layout.csl"] = gen_layout(graph, routes);

    // Generate run script
    sources["run.py"] = generate_run_script(graph);

    return sources;
}

std::string CSLCodeGenerator::gen_elementwise(const ir::Operation& op) {
    std::map<std::string, std::string> params;

    params["OP_NAME"] = op.name();
    params["TIMESTAMP"] = current_timestamp();
    params["TILE_SIZE"] = std::to_string(op.output(0).numel());
    params["DTYPE"] = dtype_to_csl(op.output(0).dtype());

    // Map operation to CSL expression
    switch (op.type()) {
        case ir::OpType::ADD:
            params["OPERATION_EXPR"] = "input_a[i] + input_b[i]";
            return templates_.instantiate(CSLTemplate::ELEMENTWISE_BINARY, params);
        case ir::OpType::MUL:
            params["OPERATION_EXPR"] = "input_a[i] * input_b[i]";
            return templates_.instantiate(CSLTemplate::ELEMENTWISE_BINARY, params);
        case ir::OpType::RELU:
            params["OPERATION_EXPR"] = "if (input[i] > 0.0) input[i] else 0.0";
            return templates_.instantiate(CSLTemplate::ELEMENTWISE_UNARY, params);
        case ir::OpType::SIGMOID:
            params["OPERATION_EXPR"] = "1.0 / (1.0 + @exp(-input[i]))";
            return templates_.instantiate(CSLTemplate::ELEMENTWISE_UNARY, params);
        default:
            throw std::runtime_error("Unknown elementwise op");
    }
}

}  // namespace pyflame::backend
```

---

## 9. Future Enhancements

### 9.1 Planned Improvements

1. **Kernel Fusion**: Automatically fuse multiple elementwise ops into single kernel
2. **Auto-Tuning**: Test multiple tile sizes and select best performing
3. **Caching**: Cache compiled kernels indexed by operation signature
4. **Debug Mode**: Generate kernels with bounds checking and logging
5. **Profiling**: Instrument generated code for performance analysis

### 9.2 Research Directions

1. **Polyhedral Compilation**: Use polyhedral model for loop optimization
2. **ML-Guided Generation**: Train model to predict optimal tiling/routing
3. **Progressive Lowering**: Multi-stage lowering for better optimization opportunities

---

## 10. References

1. Cerebras CSL Language Guide: https://sdk.cerebras.net/csl/language
2. Cerebras SDK Documentation: https://sdk.cerebras.net/
3. MLIR Project: https://mlir.llvm.org/
4. Polyhedral Compilation: https://polly.llvm.org/

---

*Document Version: 1.0*
*Authors: PyFlame Team*
*Last Updated: January 10, 2026*
