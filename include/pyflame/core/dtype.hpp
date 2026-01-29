#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace pyflame {

/// Supported data types for tensors
enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int16 = 4,
    Int8 = 5,
    Bool = 6,
    Float64 = 7,
    Int64 = 8,
};

/// Get the size in bytes of a data type
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int32: return 4;
        case DType::Int16: return 2;
        case DType::Int8: return 1;
        case DType::Bool: return 1;
        case DType::Float64: return 8;
        case DType::Int64: return 8;
        default: throw std::runtime_error("Unknown dtype");
    }
}

/// Get the name of a data type
inline std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32: return "int32";
        case DType::Int16: return "int16";
        case DType::Int8: return "int8";
        case DType::Bool: return "bool";
        case DType::Float64: return "float64";
        case DType::Int64: return "int64";
        default: return "unknown";
    }
}

/// Check if dtype is a floating point type
inline bool dtype_is_floating(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 ||
           dtype == DType::BFloat16 || dtype == DType::Float64;
}

/// Check if dtype is an integer type
inline bool dtype_is_integer(DType dtype) {
    return dtype == DType::Int32 || dtype == DType::Int16 ||
           dtype == DType::Int8 || dtype == DType::Int64;
}

/// Convert dtype to CSL type string
inline std::string dtype_to_csl(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "f32";
        case DType::Float16: return "f16";
        case DType::BFloat16: return "bf16";
        case DType::Int32: return "i32";
        case DType::Int16: return "i16";
        case DType::Int8: return "i8";
        case DType::Bool: return "bool";
        case DType::Float64: return "f64";
        case DType::Int64: return "i64";
        default: throw std::runtime_error("Unknown dtype for CSL");
    }
}

/// Stream output for DType
inline std::ostream& operator<<(std::ostream& os, DType dtype) {
    return os << dtype_name(dtype);
}

}  // namespace pyflame
