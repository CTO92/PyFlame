#pragma once

#include <string>
#include <map>
#include <filesystem>
#include <optional>

#include "pyflame/ir/graph.hpp"
#include "pyflame/core/layout.hpp"

namespace pyflame::backend {

/// Result of CSL code generation
struct CodeGenResult {
    bool success = false;
    std::filesystem::path output_dir;
    std::vector<std::filesystem::path> generated_files;
    std::string error_message;

    // Generated source code (for debugging)
    std::map<std::string, std::string> sources;
};

/// Options for code generation
struct CodeGenOptions {
    std::filesystem::path output_dir = "./pyflame_csl_output";
    std::string target = "wse2";  // "wse2" or "wse3"
    bool optimize = true;
    bool generate_debug_info = false;
    bool emit_comments = true;
    int fabric_width = 0;   // 0 = auto
    int fabric_height = 0;  // 0 = auto

    // Runtime configuration
    // For on-premises: "localhost:9000" or IP of your CS-2/CS-3
    // For Cerebras Cloud: Use the endpoint provided by your cloud instance
    std::string runtime_address = "";  // Empty = use environment variable or default
};

/// CSL Code Generator
/// Transforms a PyFlame computation graph into CSL source code
class CSLCodeGenerator {
public:
    CSLCodeGenerator();
    ~CSLCodeGenerator();

    /// Generate CSL code from an IR graph
    CodeGenResult generate(
        const ir::Graph& graph,
        const CodeGenOptions& options = CodeGenOptions{}
    );

    /// Generate CSL code and return as strings (for debugging)
    std::map<std::string, std::string> generate_source(
        const ir::Graph& graph,
        const CodeGenOptions& options = CodeGenOptions{}
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// CSL Template manager
class CSLTemplates {
public:
    static CSLTemplates& instance();

    /// Get template content for an operation
    std::string get_template(ir::OpType op) const;

    /// Get layout template
    std::string get_layout_template() const;

    /// Instantiate a template with parameters
    std::string instantiate(
        const std::string& tmpl,
        const std::map<std::string, std::string>& params
    ) const;

private:
    CSLTemplates();
    std::map<ir::OpType, std::string> templates_;
    std::string layout_template_;
};

}  // namespace pyflame::backend
