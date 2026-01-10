// CSL Emitter - Low-level CSL code emission utilities

#include "pyflame/backend/csl_codegen.hpp"
#include <sstream>
#include <iomanip>

namespace pyflame::backend {

/// Helper class for building CSL source code
class CSLEmitter {
public:
    CSLEmitter() : indent_level_(0) {}

    // Indentation control
    void indent() { ++indent_level_; }
    void dedent() { if (indent_level_ > 0) --indent_level_; }

    // Emit with current indentation
    CSLEmitter& emit(const std::string& code) {
        for (int i = 0; i < indent_level_; ++i) {
            ss_ << "    ";
        }
        ss_ << code;
        return *this;
    }

    CSLEmitter& emitln(const std::string& code = "") {
        emit(code);
        ss_ << "\n";
        return *this;
    }

    CSLEmitter& newline() {
        ss_ << "\n";
        return *this;
    }

    // Comments
    CSLEmitter& comment(const std::string& text) {
        return emitln("// " + text);
    }

    CSLEmitter& block_comment(const std::string& text) {
        emitln("/*");
        emit(" * " + text);
        return emitln(" */");
    }

    // Declarations
    CSLEmitter& param(const std::string& name, const std::string& type,
                      const std::string& default_value = "") {
        std::string line = "param " + name + ": " + type;
        if (!default_value.empty()) {
            line += " = " + default_value;
        }
        return emitln(line + ";");
    }

    CSLEmitter& constant(const std::string& name, const std::string& type,
                         const std::string& value) {
        return emitln("const " + name + ": " + type + " = " + value + ";");
    }

    CSLEmitter& variable(const std::string& name, const std::string& type,
                         const std::string& init = "") {
        std::string line = "var " + name + ": " + type;
        if (!init.empty()) {
            line += " = " + init;
        }
        return emitln(line + ";");
    }

    CSLEmitter& export_symbol(const std::string& name, const std::string& type,
                              bool input = true) {
        return emitln("export var " + name + " = &" + name + ";");
    }

    // Control structures
    CSLEmitter& task_begin(const std::string& name, const std::string& ret_type = "void") {
        emitln("task " + name + "() " + ret_type + " {");
        indent();
        return *this;
    }

    CSLEmitter& task_end() {
        dedent();
        return emitln("}");
    }

    CSLEmitter& for_range(const std::string& var, const std::string& type,
                          const std::string& start, const std::string& end) {
        emitln("for (@range(" + type + ", " + start + ", " + end + ")) |" + var + "| {");
        indent();
        return *this;
    }

    CSLEmitter& end_block() {
        dedent();
        return emitln("}");
    }

    CSLEmitter& if_begin(const std::string& condition) {
        emitln("if (" + condition + ") {");
        indent();
        return *this;
    }

    CSLEmitter& else_begin() {
        dedent();
        emitln("} else {");
        indent();
        return *this;
    }

    // Expressions
    CSLEmitter& assign(const std::string& lhs, const std::string& rhs) {
        return emitln(lhs + " = " + rhs + ";");
    }

    CSLEmitter& call(const std::string& func, const std::vector<std::string>& args = {}) {
        std::string arg_str;
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) arg_str += ", ";
            arg_str += args[i];
        }
        return emitln(func + "(" + arg_str + ");");
    }

    // Comptime block
    CSLEmitter& comptime_begin() {
        emitln("comptime {");
        indent();
        return *this;
    }

    CSLEmitter& comptime_end() {
        dedent();
        return emitln("}");
    }

    // Import
    CSLEmitter& import_module(const std::string& name, const std::string& module,
                              const std::string& params = "") {
        if (params.empty()) {
            return emitln("const " + name + " = @import_module(\"" + module + "\");");
        } else {
            return emitln("const " + name + " = @import_module(\"" + module + "\", " + params + ");");
        }
    }

    // Get result
    std::string str() const { return ss_.str(); }

    // Clear
    void clear() {
        ss_.str("");
        ss_.clear();
        indent_level_ = 0;
    }

private:
    std::ostringstream ss_;
    int indent_level_;
};

// Generate CSL for an elementwise binary operation
std::string generate_elementwise_binary_csl(
    const std::string& op_symbol,
    const std::string& dtype,
    int64_t tile_size,
    int pe_width,
    int pe_height
) {
    CSLEmitter e;

    e.comment("PyFlame Generated - Elementwise Binary Operation");
    e.newline();

    // Params
    e.emitln("param memcpy = @import_module(\"<memcpy/get_params>\", .{");
    e.indent();
    e.emitln(".width = " + std::to_string(pe_width) + ",");
    e.emitln(".height = " + std::to_string(pe_height) + ",");
    e.dedent();
    e.emitln("});");
    e.newline();

    e.import_module("sys_mod", "<memcpy_multi/memcpy>", "memcpy");
    e.newline();

    // Constants
    e.constant("TILE_SIZE", "u32", std::to_string(tile_size));
    e.newline();

    // Arrays
    e.emitln("var input_a: [TILE_SIZE]" + dtype + " = @zeros([TILE_SIZE]" + dtype + ");");
    e.emitln("var input_b: [TILE_SIZE]" + dtype + " = @zeros([TILE_SIZE]" + dtype + ");");
    e.emitln("var output: [TILE_SIZE]" + dtype + " = @zeros([TILE_SIZE]" + dtype + ");");
    e.newline();

    // Exports
    e.emitln("export var input_a_ptr = &input_a;");
    e.emitln("export var input_b_ptr = &input_b;");
    e.emitln("export var output_ptr = &output;");
    e.newline();

    // Task
    e.task_begin("compute");
    e.for_range("i", "u32", "0", "TILE_SIZE");
    e.assign("output[i]", "input_a[i] " + op_symbol + " input_b[i]");
    e.end_block();
    e.call("sys_mod.unblock_cmd_stream");
    e.task_end();
    e.newline();

    // Comptime
    e.comptime_begin();
    e.call("@bind_local_task_to_color", {"compute", "sys_mod.LAUNCH"});
    e.comptime_end();

    return e.str();
}

// Generate CSL for an elementwise unary operation
std::string generate_elementwise_unary_csl(
    const std::string& op_expr,  // e.g., "@max(0.0, input[i])" for relu
    const std::string& dtype,
    int64_t tile_size,
    int pe_width,
    int pe_height
) {
    CSLEmitter e;

    e.comment("PyFlame Generated - Elementwise Unary Operation");
    e.newline();

    e.emitln("param memcpy = @import_module(\"<memcpy/get_params>\", .{");
    e.indent();
    e.emitln(".width = " + std::to_string(pe_width) + ",");
    e.emitln(".height = " + std::to_string(pe_height) + ",");
    e.dedent();
    e.emitln("});");
    e.newline();

    e.import_module("sys_mod", "<memcpy_multi/memcpy>", "memcpy");
    e.newline();

    e.constant("TILE_SIZE", "u32", std::to_string(tile_size));
    e.newline();

    e.emitln("var input: [TILE_SIZE]" + dtype + " = @zeros([TILE_SIZE]" + dtype + ");");
    e.emitln("var output: [TILE_SIZE]" + dtype + " = @zeros([TILE_SIZE]" + dtype + ");");
    e.newline();

    e.emitln("export var input_ptr = &input;");
    e.emitln("export var output_ptr = &output;");
    e.newline();

    e.task_begin("compute");
    e.for_range("i", "u32", "0", "TILE_SIZE");
    e.assign("output[i]", op_expr);
    e.end_block();
    e.call("sys_mod.unblock_cmd_stream");
    e.task_end();
    e.newline();

    e.comptime_begin();
    e.call("@bind_local_task_to_color", {"compute", "sys_mod.LAUNCH"});
    e.comptime_end();

    return e.str();
}

}  // namespace pyflame::backend
