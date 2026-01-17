#include "pyflame/backend/csl_codegen.hpp"

#include <fstream>
#include <sstream>
#include <regex>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <set>

namespace pyflame::backend {

// ============================================================================
// Security Utilities
// ============================================================================

namespace {

/// Sanitize a path for safe display in error messages
/// Removes or masks potentially sensitive path components
std::string sanitize_path_for_error(const std::filesystem::path& path) {
    std::string result = path.filename().string();
    if (result.empty()) {
        result = "<path>";
    }
    // Only show filename, not full path
    return result;
}

/// Check if a filename is safe (no path traversal attempts)
bool is_safe_filename(const std::string& filename) {
    // Disallow empty names
    if (filename.empty()) return false;

    // Disallow path separators
    if (filename.find('/') != std::string::npos) return false;
    if (filename.find('\\') != std::string::npos) return false;

    // Disallow path traversal
    if (filename == "." || filename == "..") return false;
    if (filename.find("..") != std::string::npos) return false;

    // Disallow null bytes
    if (filename.find('\0') != std::string::npos) return false;

    // Disallow control characters
    for (char c : filename) {
        if (std::iscntrl(static_cast<unsigned char>(c))) return false;
    }

    return true;
}

/// Validate and normalize output directory path
/// Returns canonical path or throws on error
std::filesystem::path validate_output_dir(const std::filesystem::path& output_dir) {
    // Check for empty path
    if (output_dir.empty()) {
        throw std::invalid_argument("Output directory cannot be empty");
    }

    // Convert to string and check for null bytes
    std::string path_str = output_dir.string();
    if (path_str.find('\0') != std::string::npos) {
        throw std::invalid_argument("Output directory contains invalid characters");
    }

    // Check for obvious path traversal attempts in the raw path
    if (path_str.find("..") != std::string::npos) {
        // Could be legitimate (e.g., ../build), but we're conservative
        // Allow if it resolves to a valid absolute path
    }

    // Create directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create output directory");
    }

    // Get canonical (absolute, normalized) path
    auto canonical = std::filesystem::canonical(output_dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to resolve output directory path");
    }

    // Verify it's a directory
    if (!std::filesystem::is_directory(canonical, ec) || ec) {
        throw std::runtime_error("Output path is not a directory");
    }

    return canonical;
}

/// Validate that a file path is within the allowed directory
void validate_file_in_dir(
    const std::filesystem::path& filepath,
    const std::filesystem::path& allowed_dir
) {
    std::error_code ec;

    // Get canonical path of the file's parent
    auto file_canonical = std::filesystem::weakly_canonical(filepath, ec);
    if (ec) {
        throw std::runtime_error("Invalid output file path");
    }

    // Check that file is within allowed directory
    auto relative = std::filesystem::relative(file_canonical, allowed_dir, ec);
    if (ec) {
        throw std::runtime_error("Output file path validation failed");
    }

    // Check for path traversal (relative path should not start with ..)
    std::string rel_str = relative.string();
    if (rel_str.substr(0, 2) == ".." || rel_str.find("/../") != std::string::npos) {
        throw std::runtime_error("Output file path escapes allowed directory");
    }
}

/// Maximum allowed length for template parameter values
constexpr size_t MAX_PARAM_VALUE_LENGTH = 256;

/// Check if a character is a common Unicode homoglyph that could bypass ASCII checks
/// Returns the ASCII equivalent if it's a homoglyph, or the original char if not
char normalize_homoglyph(unsigned char c) {
    // Common Cyrillic/Greek homoglyphs in extended ASCII range
    // Note: Full Unicode homoglyph detection requires UTF-8 parsing
    // This catches the most common single-byte attacks
    return c;  // For single-byte chars, they're already ASCII-comparable
}

/// Validate a template parameter value to prevent CSL code injection
/// Uses a strict allowlist approach - only specific safe characters allowed
bool is_safe_template_value(const std::string& value) {
    if (value.empty()) return false;

    // Security: Limit value length to prevent buffer issues and DoS
    if (value.length() > MAX_PARAM_VALUE_LENGTH) return false;

    // Security: Check for null bytes which could truncate strings
    if (value.find('\0') != std::string::npos) return false;

    // Security: Check for non-ASCII characters (potential homoglyph attack)
    for (unsigned char c : value) {
        if (c > 127) {
            return false;  // Reject any non-ASCII
        }
    }

    // Strict character allowlist for template values
    // Security: Removed space and colon to prevent code injection patterns
    for (char c : value) {
        unsigned char uc = static_cast<unsigned char>(c);
        // Allow: alphanumeric, underscore, period, hyphen
        // Also allow brackets and commas for array syntax
        // Note: Removed ':' and ' ' to prevent injection patterns like "type: value"
        bool allowed = std::isalnum(uc) ||
                       c == '_' || c == '.' || c == '-' ||
                       c == '[' || c == ']' || c == ',';
        if (!allowed) {
            return false;
        }
    }

    // Additional check: disallow leading/trailing special chars
    if (!value.empty()) {
        char first = value.front();
        char last = value.back();
        if (!std::isalnum(static_cast<unsigned char>(first)) &&
            first != '_' && first != '[') {
            return false;
        }
        if (!std::isalnum(static_cast<unsigned char>(last)) &&
            last != '_' && last != ']') {
            return false;
        }
    }

    // Disallow suspicious patterns that could be CSL code injection
    static const std::vector<std::string> dangerous_patterns = {
        "@",     // CSL builtins
        "//",    // Comments
        "/*",    // Block comments
        "*/",    // Block comment end
        ";",     // Statement separator
        "{",     // Code blocks
        "}",     // Code block end
        "\\",    // Escape sequences
        "\"",    // String literals
        "'",     // Char literals
        "`",     // Template literals
        "$",     // Variable interpolation
        "#",     // Preprocessor
        "&&",    // Logical operators (could enable injection)
        "||",    // Logical operators
        "==",    // Comparison (may indicate complex expressions)
        "!=",    // Comparison
        "<<",    // Bit shift / stream (could be misused)
        ">>",    // Bit shift / stream
        "..",    // Range operator (could enable traversal)
        "fn ",   // Function declaration
        "var ",  // Variable declaration
        "const ",// Const declaration
        "pub ",  // Public declaration
        "import",// Import statement
        "export",// Export statement
    };

    for (const auto& pattern : dangerous_patterns) {
        if (value.find(pattern) != std::string::npos) {
            return false;
        }
    }

    return true;
}

/// Validate all template parameters before substitution
void validate_template_params(const std::map<std::string, std::string>& params) {
    // List of allowed parameter names (strict whitelist)
    static const std::set<std::string> allowed_params = {
        "TIMESTAMP", "DTYPE", "TILE_SIZE", "PE_WIDTH", "PE_HEIGHT",
        "M", "N", "K"  // Matrix dimensions for matmul
    };

    // Security: Limit total number of parameters
    constexpr size_t MAX_PARAMS = 20;
    if (params.size() > MAX_PARAMS) {
        throw std::invalid_argument(
            "Too many template parameters (" + std::to_string(params.size()) +
            "), maximum allowed: " + std::to_string(MAX_PARAMS)
        );
    }

    for (const auto& [key, value] : params) {
        // Validate parameter name is in whitelist
        if (allowed_params.find(key) == allowed_params.end()) {
            throw std::invalid_argument(
                "Unknown template parameter: '" + key + "'. "
                "Only predefined parameters are allowed for security."
            );
        }

        // Validate parameter value is safe
        if (!is_safe_template_value(value)) {
            throw std::invalid_argument(
                "Invalid template parameter value for '" + key + "': "
                "contains unsafe characters or patterns. "
                "Only alphanumeric characters, underscores, periods, hyphens, "
                "colons, spaces, and array syntax are allowed."
            );
        }
    }
}

}  // anonymous namespace

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
    // Security: Validate all template parameters before substitution
    // to prevent CSL code injection attacks
    validate_template_params(params);

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
        // Validate and normalize output directory (prevents path traversal)
        auto canonical_output_dir = validate_output_dir(options.output_dir);
        result.output_dir = canonical_output_dir;

        // Generate source
        auto sources = generate_source(graph, options);
        result.sources = sources;

        // Write files with security validation
        for (const auto& [filename, content] : sources) {
            // Validate filename is safe (no path traversal)
            if (!is_safe_filename(filename)) {
                result.success = false;
                result.error_message = "Invalid filename in generated sources";
                return result;
            }

            auto filepath = canonical_output_dir / filename;

            // Double-check file stays within output directory
            validate_file_in_dir(filepath, canonical_output_dir);

            std::ofstream file(filepath);
            if (!file) {
                result.success = false;
                // Sanitize path in error message to avoid information leakage
                result.error_message = "Failed to write file: " + sanitize_path_for_error(filepath);
                return result;
            }
            file << content;
            result.generated_files.push_back(filepath);
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.success = false;
        // Use generic error message to avoid leaking sensitive information
        result.error_message = "Code generation failed: " + std::string(e.what());
    }

    return result;
}

}  // namespace pyflame::backend
