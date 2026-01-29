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

    // Convolution operations (Phase 2)
    CONV1D = 510,
    CONV2D = 511,
    CONV3D = 512,
    CONV_TRANSPOSE2D = 513,

    // Pooling operations (Phase 2)
    MAX_POOL1D = 520,
    MAX_POOL2D = 521,
    MAX_POOL3D = 522,
    AVG_POOL1D = 523,
    AVG_POOL2D = 524,
    AVG_POOL3D = 525,
    ADAPTIVE_AVG_POOL2D = 526,
    ADAPTIVE_AVG_POOL1D = 527,

    // Normalization operations (Phase 2)
    BATCH_NORM = 530,
    LAYER_NORM = 531,
    GROUP_NORM = 532,
    INSTANCE_NORM = 533,

    // Dropout (Phase 2)
    DROPOUT = 540,

    // Embedding (Phase 2)
    EMBEDDING = 545,
    GATHER = 546,

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
    LESS = 806,        // Same as LT, for clarity

    // Conditional operations
    WHERE = 810,       // Ternary selection
    MAXIMUM = 811,     // Element-wise maximum
    MINIMUM = 812,     // Element-wise minimum
    CLAMP = 813,       // Clamp values to range

    // Loss function operations (Phase 2)
    NLL_LOSS = 850,
    CROSS_ENTROPY_LOSS = 851,
    MSE_LOSS = 852,
    L1_LOSS = 853,
    BCE_LOSS = 854,
    BCE_WITH_LOGITS_LOSS = 855,
    KL_DIV_LOSS = 856,
    SMOOTH_L1_LOSS = 857,

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
        case OpType::CONV1D: return "conv1d";
        case OpType::CONV2D: return "conv2d";
        case OpType::CONV3D: return "conv3d";
        case OpType::CONV_TRANSPOSE2D: return "conv_transpose2d";
        case OpType::MAX_POOL1D: return "max_pool1d";
        case OpType::MAX_POOL2D: return "max_pool2d";
        case OpType::MAX_POOL3D: return "max_pool3d";
        case OpType::AVG_POOL1D: return "avg_pool1d";
        case OpType::AVG_POOL2D: return "avg_pool2d";
        case OpType::AVG_POOL3D: return "avg_pool3d";
        case OpType::ADAPTIVE_AVG_POOL2D: return "adaptive_avg_pool2d";
        case OpType::ADAPTIVE_AVG_POOL1D: return "adaptive_avg_pool1d";
        case OpType::BATCH_NORM: return "batch_norm";
        case OpType::LAYER_NORM: return "layer_norm";
        case OpType::GROUP_NORM: return "group_norm";
        case OpType::INSTANCE_NORM: return "instance_norm";
        case OpType::DROPOUT: return "dropout";
        case OpType::EMBEDDING: return "embedding";
        case OpType::GATHER: return "gather";
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
        case OpType::LESS: return "less";
        case OpType::WHERE: return "where";
        case OpType::MAXIMUM: return "maximum";
        case OpType::MINIMUM: return "minimum";
        case OpType::CLAMP: return "clamp";
        case OpType::NLL_LOSS: return "nll_loss";
        case OpType::CROSS_ENTROPY_LOSS: return "cross_entropy_loss";
        case OpType::MSE_LOSS: return "mse_loss";
        case OpType::L1_LOSS: return "l1_loss";
        case OpType::BCE_LOSS: return "bce_loss";
        case OpType::BCE_WITH_LOGITS_LOSS: return "bce_with_logits_loss";
        case OpType::KL_DIV_LOSS: return "kl_div_loss";
        case OpType::SMOOTH_L1_LOSS: return "smooth_l1_loss";
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
