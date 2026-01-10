#pragma once

/// @file pyflame.hpp
/// @brief Main include file for PyFlame library
///
/// @note PRE-RELEASE ALPHA 1.0
///       This software is in early development and is not yet ready for production use.
///       APIs may change without notice.

// Core
#include "pyflame/core/dtype.hpp"
#include "pyflame/core/layout.hpp"
#include "pyflame/core/allocator.hpp"
#include "pyflame/core/tensor.hpp"
#include "pyflame/core/tensor_impl.hpp"

// IR
#include "pyflame/ir/op_type.hpp"
#include "pyflame/ir/node.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/ir/shape_inference.hpp"

// Backend
#include "pyflame/backend/csl_codegen.hpp"

/// @namespace pyflame
/// @brief PyFlame - Native deep learning framework for Cerebras WSE
///        Pre-Release Alpha 1.0
namespace pyflame {

/// Library version (Pre-Release Alpha 1.0)
constexpr const char* VERSION = "1.0.0-alpha";
constexpr const char* VERSION_SUFFIX = "-alpha";
constexpr const char* RELEASE_STATUS = "Pre-Release Alpha";
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

/// Check if this is a pre-release version
constexpr bool IS_PRERELEASE = true;

}  // namespace pyflame
