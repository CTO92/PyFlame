#include "pyflame/backend/csl_codegen.hpp"

#include <fstream>
#include <sstream>
#include <regex>
#include <chrono>
#include <iomanip>

namespace pyflame::backend {

// ============================================================================
// CSL Templates
// ============================================================================

CSLTemplates& CSLTemplates::instance() {
    static CSLTemplates instance;
    return instance;
}

CSLTemplates::CSLTemplates() {
    // Elementwise binary template
    templates_[ir::OpType::ADD] = R"(
// PyFlame Generated - Elementwise Add
// Generated: {{TIMESTAMP}}

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_WIDTH}},
    .height = {{PE_HEIGHT}},
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);

const TILE_SIZE: u32 = {{TILE_SIZE}};

var input_a: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
var input_b: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
var output: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});

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
)";

    templates_[ir::OpType::MUL] = R"(
// PyFlame Generated - Elementwise Multiply
// Generated: {{TIMESTAMP}}

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_WIDTH}},
    .height = {{PE_HEIGHT}},
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);

const TILE_SIZE: u32 = {{TILE_SIZE}};

var input_a: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
var input_b: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
var output: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});

export var input_a_ptr = &input_a;
export var input_b_ptr = &input_b;
export var output_ptr = &output;

task compute() void {
    for (@range(u32, 0, TILE_SIZE)) |i| {
        output[i] = input_a[i] * input_b[i];
    }
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task_to_color(compute, sys_mod.LAUNCH);
}
)";

    templates_[ir::OpType::RELU] = R"(
// PyFlame Generated - ReLU Activation
// Generated: {{TIMESTAMP}}

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_WIDTH}},
    .height = {{PE_HEIGHT}},
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);

const TILE_SIZE: u32 = {{TILE_SIZE}};

var input: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
var output: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});

export var input_ptr = &input;
export var output_ptr = &output;

task compute() void {
    for (@range(u32, 0, TILE_SIZE)) |i| {
        const val = input[i];
        output[i] = if (val > 0.0) val else 0.0;
    }
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task_to_color(compute, sys_mod.LAUNCH);
}
)";

    templates_[ir::OpType::SIGMOID] = R"(
// PyFlame Generated - Sigmoid Activation
// Generated: {{TIMESTAMP}}

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_WIDTH}},
    .height = {{PE_HEIGHT}},
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);
const math = @import_module("<math>");

const TILE_SIZE: u32 = {{TILE_SIZE}};

var input: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
var output: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});

export var input_ptr = &input;
export var output_ptr = &output;

task compute() void {
    for (@range(u32, 0, TILE_SIZE)) |i| {
        output[i] = 1.0 / (1.0 + math.exp(-input[i]));
    }
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task_to_color(compute, sys_mod.LAUNCH);
}
)";

    templates_[ir::OpType::SUM] = R"(
// PyFlame Generated - Reduction Sum
// Generated: {{TIMESTAMP}}

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_WIDTH}},
    .height = {{PE_HEIGHT}},
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);

const TILE_SIZE: u32 = {{TILE_SIZE}};

var input: [TILE_SIZE]{{DTYPE}} = @zeros([TILE_SIZE]{{DTYPE}});
var output: [1]{{DTYPE}} = @zeros([1]{{DTYPE}});

export var input_ptr = &input;
export var output_ptr = &output;

task compute() void {
    var sum: {{DTYPE}} = 0.0;
    for (@range(u32, 0, TILE_SIZE)) |i| {
        sum += input[i];
    }
    output[0] = sum;
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task_to_color(compute, sys_mod.LAUNCH);
}
)";

    templates_[ir::OpType::MATMUL] = R"(
// PyFlame Generated - Matrix Multiplication (Single PE)
// Generated: {{TIMESTAMP}}
// Computes C = A @ B where A:[M,K], B:[K,N], C:[M,N]

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_WIDTH}},
    .height = {{PE_HEIGHT}},
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);

const M: u32 = {{M}};
const K: u32 = {{K}};
const N: u32 = {{N}};

var A: [M][K]{{DTYPE}} = @zeros([M][K]{{DTYPE}});
var B: [K][N]{{DTYPE}} = @zeros([K][N]{{DTYPE}});
var C: [M][N]{{DTYPE}} = @zeros([M][N]{{DTYPE}});

export var A_ptr = &A;
export var B_ptr = &B;
export var C_ptr = &C;

task compute() void {
    for (@range(u32, 0, M)) |i| {
        for (@range(u32, 0, N)) |j| {
            var sum: {{DTYPE}} = 0.0;
            for (@range(u32, 0, K)) |k| {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task_to_color(compute, sys_mod.LAUNCH);
}
)";

    // Layout template
    layout_template_ = R"(
// PyFlame Generated Layout
// Generated: {{TIMESTAMP}}

param memcpy_params: comptime_struct;

const memcpy = @import_module("<memcpy_multi/memcpy>", memcpy_params);

layout {
    @set_rectangle({{PE_WIDTH}}, {{PE_HEIGHT}});

    // Configure PE at (0, 0)
    @set_tile_code(0, 0, "pe_program.csl", .{
        .memcpy_params = memcpy.get_params(0),
    });

    // Export symbols for host access
    @export_name("input_a_ptr", [*]{{DTYPE}}, true);
    @export_name("input_b_ptr", [*]{{DTYPE}}, true);
    @export_name("output_ptr", [*]{{DTYPE}}, false);
}
)";
}

std::string CSLTemplates::get_template(ir::OpType op) const {
    auto it = templates_.find(op);
    if (it == templates_.end()) {
        throw std::runtime_error("No CSL template for operation: " + ir::op_type_name(op));
    }
    return it->second;
}

std::string CSLTemplates::get_layout_template() const {
    return layout_template_;
}

std::string CSLTemplates::instantiate(
    const std::string& tmpl,
    const std::map<std::string, std::string>& params
) const {
    std::string result = tmpl;

    // Replace {{PARAM}} patterns
    for (const auto& [key, value] : params) {
        std::string pattern = "{{" + key + "}}";
        size_t pos = 0;
        while ((pos = result.find(pattern, pos)) != std::string::npos) {
            result.replace(pos, pattern.length(), value);
            pos += value.length();
        }
    }

    return result;
}

// ============================================================================
// CSL Code Generator Implementation
// ============================================================================

struct CSLCodeGenerator::Impl {
    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::ostringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    std::string dtype_to_csl(DType dtype) const {
        switch (dtype) {
            case DType::Float32: return "f32";
            case DType::Float16: return "f16";
            case DType::Int32: return "i32";
            case DType::Int16: return "i16";
            default: return "f32";
        }
    }

    std::map<std::string, std::string> generate_params(
        const ir::Node& node,
        const CodeGenOptions& options
    ) const {
        std::map<std::string, std::string> params;

        params["TIMESTAMP"] = get_timestamp();
        params["DTYPE"] = dtype_to_csl(node.dtype());
        params["TILE_SIZE"] = std::to_string(node.numel());
        params["PE_WIDTH"] = std::to_string(options.fabric_width > 0 ? options.fabric_width : 1);
        params["PE_HEIGHT"] = std::to_string(options.fabric_height > 0 ? options.fabric_height : 1);

        // For matmul, extract dimensions
        if (node.op_type() == ir::OpType::MATMUL) {
            auto shape = node.shape();
            if (shape.size() >= 2) {
                params["M"] = std::to_string(shape[shape.size() - 2]);
                params["N"] = std::to_string(shape[shape.size() - 1]);

                // Get K from first input
                if (!node.inputs().empty()) {
                    auto input_shape = node.inputs()[0]->shape();
                    if (input_shape.size() >= 2) {
                        params["K"] = std::to_string(input_shape[input_shape.size() - 1]);
                    }
                }
            }
        }

        return params;
    }

    std::string generate_pe_program(
        const ir::Graph& graph,
        const CodeGenOptions& options
    ) const {
        auto& templates = CSLTemplates::instance();

        // For now, generate code for the last operation
        // Full implementation would handle the entire graph
        auto topo = graph.topological_order();

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const auto& node = *it;
            if (node->is_operation()) {
                try {
                    std::string tmpl = templates.get_template(node->op_type());
                    auto params = generate_params(*node, options);
                    return templates.instantiate(tmpl, params);
                } catch (...) {
                    // No template for this op, try next
                    continue;
                }
            }
        }

        return "// No operations to generate\n";
    }

    std::string generate_layout(
        const ir::Graph& graph,
        const CodeGenOptions& options
    ) const {
        auto& templates = CSLTemplates::instance();
        std::string tmpl = templates.get_layout_template();

        // Find the output dtype
        DType dtype = DType::Float32;
        if (!graph.outputs().empty()) {
            dtype = graph.outputs()[0]->dtype();
        }

        std::map<std::string, std::string> params;
        params["TIMESTAMP"] = get_timestamp();
        params["DTYPE"] = dtype_to_csl(dtype);
        params["PE_WIDTH"] = std::to_string(options.fabric_width > 0 ? options.fabric_width : 1);
        params["PE_HEIGHT"] = std::to_string(options.fabric_height > 0 ? options.fabric_height : 1);

        return templates.instantiate(tmpl, params);
    }

    std::string generate_run_script(
        const ir::Graph& graph,
        const CodeGenOptions& options
    ) const {
        std::ostringstream ss;
        ss << "#!/usr/bin/env python3\n";
        ss << "# PyFlame Generated Run Script\n";
        ss << "# Generated: " << get_timestamp() << "\n\n";

        ss << "import numpy as np\n";
        ss << "from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType\n\n";

        ss << "# Configuration\n";
        ss << "ELF_PATH = 'out.elf'\n";
        ss << "FABRIC_WIDTH = " << (options.fabric_width > 0 ? options.fabric_width : 1) << "\n";
        ss << "FABRIC_HEIGHT = " << (options.fabric_height > 0 ? options.fabric_height : 1) << "\n\n";

        ss << "def main():\n";
        ss << "    import os\n\n";
        ss << "    # Runtime address configuration:\n";
        ss << "    # - On-premises: 'localhost:9000' or your CS-2/CS-3 IP address\n";
        ss << "    # - Cerebras Cloud: Use the endpoint URL from your cloud instance\n";
        ss << "    # Set CEREBRAS_RUNTIME_ADDRESS environment variable or modify below\n";
        if (options.runtime_address.empty()) {
            ss << "    cmaddr = os.environ.get('CEREBRAS_RUNTIME_ADDRESS', 'localhost:9000')\n\n";
        } else {
            ss << "    cmaddr = '" << options.runtime_address << "'\n\n";
        }
        ss << "    # Initialize runtime\n";
        ss << "    runner = SdkRuntime(ELF_PATH, cmaddr=cmaddr)\n\n";

        ss << "    # Load and run\n";
        ss << "    runner.load()\n";
        ss << "    runner.run()\n\n";

        ss << "    # Get results\n";
        ss << "    # output = runner.memcpy_d2h(...)\n\n";

        ss << "    runner.stop()\n";
        ss << "    print('Done!')\n\n";

        ss << "if __name__ == '__main__':\n";
        ss << "    main()\n";

        return ss.str();
    }
};

CSLCodeGenerator::CSLCodeGenerator() : impl_(std::make_unique<Impl>()) {}
CSLCodeGenerator::~CSLCodeGenerator() = default;

std::map<std::string, std::string> CSLCodeGenerator::generate_source(
    const ir::Graph& graph,
    const CodeGenOptions& options
) {
    std::map<std::string, std::string> sources;

    sources["pe_program.csl"] = impl_->generate_pe_program(graph, options);
    sources["layout.csl"] = impl_->generate_layout(graph, options);
    sources["run.py"] = impl_->generate_run_script(graph, options);

    return sources;
}

CodeGenResult CSLCodeGenerator::generate(
    const ir::Graph& graph,
    const CodeGenOptions& options
) {
    CodeGenResult result;

    try {
        // Generate source
        auto sources = generate_source(graph, options);
        result.sources = sources;

        // Create output directory
        std::filesystem::create_directories(options.output_dir);
        result.output_dir = options.output_dir;

        // Write files
        for (const auto& [filename, content] : sources) {
            auto filepath = options.output_dir / filename;
            std::ofstream file(filepath);
            if (!file) {
                result.success = false;
                result.error_message = "Failed to write file: " + filepath.string();
                return result;
            }
            file << content;
            result.generated_files.push_back(filepath);
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }

    return result;
}

}  // namespace pyflame::backend
