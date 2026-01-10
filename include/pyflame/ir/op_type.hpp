#pragma once

#include <string>
#include <cstdint>

namespace pyflame::ir {

/// Types of operations in the computation graph
enum class OpType : uint16_t {
    NONE = 0,

    // Elementwise binary operations
    ADD = 100,
    SUB = 101,
    MUL = 102,
    DIV = 103,
    POW = 104,
    MAX_BINARY = 105,
    MIN_BINARY = 106,

    // Elementwise unary operations
    NEG = 200,
    ABS = 201,
    SQRT = 202,
    EXP = 203,
    LOG = 204,
    SIN = 205,
    COS = 206,
    TANH = 207,

    // Activation functions
    RELU = 300,
    SIGMOID = 301,
    GELU = 302,
    SILU = 303,
    SOFTMAX = 304,
    LOG_SOFTMAX = 305,

    // Reduction operations
    SUM = 400,
    MEAN = 401,
    MAX = 402,
    MIN = 403,
    PROD = 404,

    // Matrix operations
    MATMUL = 500,
    TRANSPOSE = 501,
    BATCH_MATMUL = 502,

    // Shape operations
    RESHAPE = 600,
    VIEW = 601,
    SQUEEZE = 602,
    UNSQUEEZE = 603,
    SLICE = 604,
    CONCAT = 605,
    STACK = 606,
    BROADCAST = 607,

    // Data movement
    COPY = 700,
    CONTIGUOUS = 701,
    LAYOUT_TRANSFORM = 702,

    // Comparison operations
    EQ = 800,
    NE = 801,
    LT = 802,
    LE = 803,
    GT = 804,
    GE = 805,

    // Special
    CONSTANT = 900,
    INPUT = 901,
    PARAMETER = 902,
};

/// Get the name of an operation type
inline std::string op_type_name(OpType op) {
    switch (op) {
        case OpType::NONE: return "none";
        case OpType::ADD: return "add";
        case OpType::SUB: return "sub";
        case OpType::MUL: return "mul";
        case OpType::DIV: return "div";
        case OpType::POW: return "pow";
        case OpType::MAX_BINARY: return "max";
        case OpType::MIN_BINARY: return "min";
        case OpType::NEG: return "neg";
        case OpType::ABS: return "abs";
        case OpType::SQRT: return "sqrt";
        case OpType::EXP: return "exp";
        case OpType::LOG: return "log";
        case OpType::SIN: return "sin";
        case OpType::COS: return "cos";
        case OpType::TANH: return "tanh";
        case OpType::RELU: return "relu";
        case OpType::SIGMOID: return "sigmoid";
        case OpType::GELU: return "gelu";
        case OpType::SILU: return "silu";
        case OpType::SOFTMAX: return "softmax";
        case OpType::LOG_SOFTMAX: return "log_softmax";
        case OpType::SUM: return "sum";
        case OpType::MEAN: return "mean";
        case OpType::MAX: return "max";
        case OpType::MIN: return "min";
        case OpType::PROD: return "prod";
        case OpType::MATMUL: return "matmul";
        case OpType::TRANSPOSE: return "transpose";
        case OpType::BATCH_MATMUL: return "batch_matmul";
        case OpType::RESHAPE: return "reshape";
        case OpType::VIEW: return "view";
        case OpType::SQUEEZE: return "squeeze";
        case OpType::UNSQUEEZE: return "unsqueeze";
        case OpType::SLICE: return "slice";
        case OpType::CONCAT: return "concat";
        case OpType::STACK: return "stack";
        case OpType::BROADCAST: return "broadcast";
        case OpType::COPY: return "copy";
        case OpType::CONTIGUOUS: return "contiguous";
        case OpType::LAYOUT_TRANSFORM: return "layout_transform";
        case OpType::EQ: return "eq";
        case OpType::NE: return "ne";
        case OpType::LT: return "lt";
        case OpType::LE: return "le";
        case OpType::GT: return "gt";
        case OpType::GE: return "ge";
        case OpType::CONSTANT: return "constant";
        case OpType::INPUT: return "input";
        case OpType::PARAMETER: return "parameter";
        default: return "unknown";
    }
}

/// Check if operation is elementwise
inline bool is_elementwise(OpType op) {
    uint16_t val = static_cast<uint16_t>(op);
    return (val >= 100 && val < 400) || (val >= 800 && val < 900);
}

/// Check if operation is a reduction
inline bool is_reduction(OpType op) {
    uint16_t val = static_cast<uint16_t>(op);
    return val >= 400 && val < 500;
}

/// Check if operation is unary
inline bool is_unary(OpType op) {
    uint16_t val = static_cast<uint16_t>(op);
    return (val >= 200 && val < 400);
}

/// Check if operation is binary
inline bool is_binary(OpType op) {
    uint16_t val = static_cast<uint16_t>(op);
    return (val >= 100 && val < 200) || (val >= 800 && val < 900);
}

/// Check if operation is an activation function
inline bool is_activation(OpType op) {
    uint16_t val = static_cast<uint16_t>(op);
    return val >= 300 && val < 400;
}

}  // namespace pyflame::ir
