# Build System Setup

**PyFlame Version:** Pre-Release Alpha 1.0
**Document Version:** 1.0
**Last Updated:** January 10, 2026
**Status:** Design Phase

> **Note:** This document is part of PyFlame Pre-Release Alpha 1.0. APIs and designs described here are subject to change.

---

## 1. Overview

PyFlame uses **CMake** as its build system, with **pybind11** for Python bindings and integration with the **Cerebras SDK** for WSE compilation. This document describes the complete build infrastructure.

### 1.1 Build Requirements

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| **CMake** | 3.18+ | For modern C++ and pybind11 support |
| **C++ Compiler** | C++17 | GCC 9+, Clang 10+, or MSVC 2019+ |
| **Python** | 3.8+ | For bindings and SDK |
| **pybind11** | 2.10+ | Python-C++ bindings |
| **Cerebras SDK** | Latest | CSL compiler and runtime |
| **Git** | 2.0+ | For dependency fetching |

### 1.2 Optional Dependencies

| Dependency | Purpose |
|------------|---------|
| **Google Test** | C++ unit testing |
| **pytest** | Python testing |
| **Doxygen** | API documentation |
| **clang-format** | Code formatting |
| **clang-tidy** | Static analysis |

---

## 2. Project Structure

```
pyflame/
├── CMakeLists.txt              # Root CMake file
├── cmake/
│   ├── PyFlameConfig.cmake.in  # Package config template
│   ├── FindCerebrasSdk.cmake   # Find Cerebras SDK
│   ├── CompilerOptions.cmake   # Compiler flags
│   ├── Dependencies.cmake      # Third-party deps
│   └── Version.cmake           # Version handling
├── include/
│   └── pyflame/                # Public headers
├── src/
│   ├── CMakeLists.txt          # Library sources
│   ├── core/                   # Core library
│   ├── ir/                     # Graph IR
│   ├── backend/                # CSL generation
│   └── runtime/                # Execution runtime
├── python/
│   ├── CMakeLists.txt          # Python bindings
│   ├── pyflame/                # Python package
│   └── bindings.cpp            # pybind11 bindings
├── tests/
│   ├── CMakeLists.txt          # Test configuration
│   ├── cpp/                    # C++ tests
│   └── python/                 # Python tests
├── examples/
│   ├── CMakeLists.txt          # Example programs
│   ├── cpp/
│   └── python/
├── third_party/
│   └── CMakeLists.txt          # Third-party deps
├── docs/
│   └── Doxyfile.in             # Documentation config
├── .github/
│   └── workflows/              # CI/CD pipelines
├── pyproject.toml              # Python package config
├── setup.py                    # Python setup (generated)
└── README.md
```

---

## 3. Root CMakeLists.txt

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

# Version from git or manual
include(cmake/Version.cmake)
determine_version()

project(PyFlame
    VERSION ${PYFLAME_VERSION_MAJOR}.${PYFLAME_VERSION_MINOR}.${PYFLAME_VERSION_PATCH}
    LANGUAGES CXX
    DESCRIPTION "Native deep learning framework for Cerebras WSE"
)

# ============================================================================
# Options
# ============================================================================

option(PYFLAME_BUILD_PYTHON "Build Python bindings" ON)
option(PYFLAME_BUILD_TESTS "Build unit tests" ON)
option(PYFLAME_BUILD_EXAMPLES "Build example programs" ON)
option(PYFLAME_BUILD_DOCS "Build documentation" OFF)
option(PYFLAME_USE_CEREBRAS_SDK "Enable Cerebras SDK integration" ON)
option(PYFLAME_ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)
option(PYFLAME_ENABLE_COVERAGE "Enable code coverage" OFF)

# ============================================================================
# CMake configuration
# ============================================================================

# Use folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ============================================================================
# Paths
# ============================================================================

set(PYFLAME_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PYFLAME_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PYFLAME_INCLUDE_DIR ${PYFLAME_SOURCE_DIR}/include)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYFLAME_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYFLAME_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PYFLAME_BINARY_DIR}/bin)

# ============================================================================
# Compiler configuration
# ============================================================================

include(cmake/CompilerOptions.cmake)
set_compiler_options()

# ============================================================================
# Dependencies
# ============================================================================

include(cmake/Dependencies.cmake)

# pybind11 for Python bindings
if(PYFLAME_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Cerebras SDK
if(PYFLAME_USE_CEREBRAS_SDK)
    include(cmake/FindCerebrasSdk.cmake)
    find_cerebras_sdk()
endif()

# Google Test for unit tests
if(PYFLAME_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ============================================================================
# Main library
# ============================================================================

add_subdirectory(src)

# ============================================================================
# Python bindings
# ============================================================================

if(PYFLAME_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ============================================================================
# Tests
# ============================================================================

if(PYFLAME_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ============================================================================
# Examples
# ============================================================================

if(PYFLAME_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Documentation
# ============================================================================

if(PYFLAME_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    configure_file(docs/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY include/pyflame
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate and install package config
configure_package_config_file(
    cmake/PyFlameConfig.cmake.in
    ${CMAKE_BINARY_DIR}/PyFlameConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PyFlame
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/PyFlameConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_BINARY_DIR}/PyFlameConfig.cmake
    ${CMAKE_BINARY_DIR}/PyFlameConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PyFlame
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "PyFlame ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "  Build Python:      ${PYFLAME_BUILD_PYTHON}")
message(STATUS "  Build tests:       ${PYFLAME_BUILD_TESTS}")
message(STATUS "  Build examples:    ${PYFLAME_BUILD_EXAMPLES}")
message(STATUS "  Build docs:        ${PYFLAME_BUILD_DOCS}")
message(STATUS "  Cerebras SDK:      ${PYFLAME_USE_CEREBRAS_SDK}")
if(CEREBRAS_SDK_FOUND)
    message(STATUS "    SDK path:        ${CEREBRAS_SDK_DIR}")
    message(STATUS "    SDK version:     ${CEREBRAS_SDK_VERSION}")
endif()
message(STATUS "")
```

---

## 4. Source Library CMake

```cmake
# src/CMakeLists.txt

# Collect source files
set(PYFLAME_CORE_SOURCES
    core/tensor.cpp
    core/tensor_impl.cpp
    core/dtype.cpp
    core/layout.cpp
    core/context.cpp
    core/allocator.cpp
)

set(PYFLAME_IR_SOURCES
    ir/graph.cpp
    ir/node.cpp
    ir/operation.cpp
    ir/shape_inference.cpp
    ir/serialization.cpp
)

set(PYFLAME_PASSES_SOURCES
    ir/passes/pass_manager.cpp
    ir/passes/constant_folding.cpp
    ir/passes/dead_code_elimination.cpp
    ir/passes/cse.cpp
    ir/passes/operator_fusion.cpp
    ir/passes/layout_optimization.cpp
)

set(PYFLAME_BACKEND_SOURCES
    backend/csl_codegen.cpp
    backend/csl_emitter.cpp
    backend/csl_templates.cpp
    backend/routing_planner.cpp
    backend/pe_assignment.cpp
    backend/csl_compiler.cpp
)

set(PYFLAME_RUNTIME_SOURCES
    runtime/executor.cpp
    runtime/transfer.cpp
    runtime/simulator_backend.cpp
    runtime/hardware_backend.cpp
)

set(PYFLAME_MEMORY_SOURCES
    memory/host_allocator.cpp
    memory/pe_memory_planner.cpp
)

set(PYFLAME_LAYOUT_SOURCES
    layout/layout_planner.cpp
    layout/layout_transform.cpp
    layout/transform_insertion.cpp
)

# ============================================================================
# Core library (static)
# ============================================================================

add_library(pyflame_core STATIC
    ${PYFLAME_CORE_SOURCES}
    ${PYFLAME_IR_SOURCES}
    ${PYFLAME_PASSES_SOURCES}
    ${PYFLAME_MEMORY_SOURCES}
    ${PYFLAME_LAYOUT_SOURCES}
)

target_include_directories(pyflame_core
    PUBLIC
        $<BUILD_INTERFACE:${PYFLAME_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(pyflame_core PUBLIC cxx_std_17)

set_target_properties(pyflame_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME pyflame_core
)

# ============================================================================
# Backend library (static, optional Cerebras integration)
# ============================================================================

add_library(pyflame_backend STATIC
    ${PYFLAME_BACKEND_SOURCES}
    ${PYFLAME_RUNTIME_SOURCES}
)

target_include_directories(pyflame_backend
    PUBLIC
        $<BUILD_INTERFACE:${PYFLAME_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pyflame_backend
    PUBLIC pyflame_core
)

target_compile_features(pyflame_backend PUBLIC cxx_std_17)

set_target_properties(pyflame_backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Link Cerebras SDK if available
if(CEREBRAS_SDK_FOUND)
    target_include_directories(pyflame_backend PRIVATE ${CEREBRAS_SDK_INCLUDE_DIR})
    target_link_libraries(pyflame_backend PRIVATE ${CEREBRAS_SDK_LIBRARIES})
    target_compile_definitions(pyflame_backend PRIVATE PYFLAME_HAS_CEREBRAS_SDK)
endif()

# ============================================================================
# Combined library (for external linking)
# ============================================================================

add_library(pyflame STATIC
    $<TARGET_OBJECTS:pyflame_core>
    $<TARGET_OBJECTS:pyflame_backend>
)

target_include_directories(pyflame
    PUBLIC
        $<BUILD_INTERFACE:${PYFLAME_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(pyflame PUBLIC cxx_std_17)

# ============================================================================
# Shared library (optional)
# ============================================================================

if(BUILD_SHARED_LIBS)
    add_library(pyflame_shared SHARED
        $<TARGET_OBJECTS:pyflame_core>
        $<TARGET_OBJECTS:pyflame_backend>
    )

    target_include_directories(pyflame_shared
        PUBLIC
            $<BUILD_INTERFACE:${PYFLAME_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    set_target_properties(pyflame_shared PROPERTIES
        OUTPUT_NAME pyflame
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    if(CEREBRAS_SDK_FOUND)
        target_link_libraries(pyflame_shared PRIVATE ${CEREBRAS_SDK_LIBRARIES})
    endif()
endif()

# ============================================================================
# CSL Templates (install as data)
# ============================================================================

# Copy CSL templates to build directory
file(GLOB CSL_TEMPLATES "${CMAKE_CURRENT_SOURCE_DIR}/backend/csl_templates/*.csl.template")
file(COPY ${CSL_TEMPLATES} DESTINATION ${CMAKE_BINARY_DIR}/share/pyflame/templates)

install(FILES ${CSL_TEMPLATES}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/pyflame/templates
)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS pyflame pyflame_core pyflame_backend
    EXPORT PyFlameTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT PyFlameTargets
    FILE PyFlameTargets.cmake
    NAMESPACE PyFlame::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PyFlame
)
```

---

## 5. Python Bindings CMake

```cmake
# python/CMakeLists.txt

# ============================================================================
# Python extension module
# ============================================================================

pybind11_add_module(_pyflame_cpp MODULE
    bindings.cpp
    bindings_tensor.cpp
    bindings_layout.cpp
    bindings_ops.cpp
    bindings_graph.cpp
    bindings_executor.cpp
)

target_link_libraries(_pyflame_cpp PRIVATE
    pyflame_core
    pyflame_backend
)

target_include_directories(_pyflame_cpp PRIVATE
    ${PYFLAME_INCLUDE_DIR}
)

# Set output directory to match Python package structure
set_target_properties(_pyflame_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pyflame
)

# ============================================================================
# Copy Python source files
# ============================================================================

file(GLOB_RECURSE PYTHON_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/pyflame/*.py"
)

foreach(PYTHON_FILE ${PYTHON_SOURCE_FILES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_FILE})
    configure_file(${PYTHON_FILE} ${CMAKE_BINARY_DIR}/python/${REL_PATH} COPYONLY)
endforeach()

# ============================================================================
# Generate __version__.py
# ============================================================================

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pyflame/_version.py.in
    ${CMAKE_BINARY_DIR}/python/pyflame/_version.py
    @ONLY
)

# ============================================================================
# Create setup.py for pip install
# ============================================================================

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_BINARY_DIR}/setup.py
    @ONLY
)

# ============================================================================
# Stub generation for type hints
# ============================================================================

find_program(STUBGEN stubgen)
if(STUBGEN)
    add_custom_command(TARGET _pyflame_cpp POST_BUILD
        COMMAND ${STUBGEN} -m _pyflame_cpp -o ${CMAKE_BINARY_DIR}/python/pyflame
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/python/pyflame
        COMMENT "Generating Python stubs"
    )
endif()

# ============================================================================
# Install Python package
# ============================================================================

# Custom target for pip install in development mode
add_custom_target(pip-install
    COMMAND ${Python3_EXECUTABLE} -m pip install -e ${CMAKE_BINARY_DIR}
    DEPENDS _pyflame_cpp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Installing PyFlame in development mode"
)

# Install to site-packages
install(DIRECTORY ${CMAKE_BINARY_DIR}/python/pyflame
    DESTINATION ${Python3_SITELIB}
    PATTERN "*.pyc" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
)
```

---

## 6. Cerebras SDK Integration

```cmake
# cmake/FindCerebrasSdk.cmake

# Find and configure the Cerebras SDK

function(find_cerebras_sdk)
    # Check environment variable first
    if(DEFINED ENV{CEREBRAS_SDK_DIR})
        set(CEREBRAS_SDK_HINT $ENV{CEREBRAS_SDK_DIR})
    elseif(DEFINED ENV{CEREBRAS_SDK_PATH})
        set(CEREBRAS_SDK_HINT $ENV{CEREBRAS_SDK_PATH})
    endif()

    # Common installation paths
    set(CEREBRAS_SDK_SEARCH_PATHS
        ${CEREBRAS_SDK_HINT}
        /opt/cerebras/sdk
        /usr/local/cerebras/sdk
        $ENV{HOME}/cerebras/sdk
        C:/cerebras/sdk
    )

    # Find cslc compiler
    find_program(CEREBRAS_CSLC
        NAMES cslc
        PATHS ${CEREBRAS_SDK_SEARCH_PATHS}
        PATH_SUFFIXES bin
    )

    # Find SDK root from compiler location
    if(CEREBRAS_CSLC)
        get_filename_component(CEREBRAS_SDK_BIN_DIR ${CEREBRAS_CSLC} DIRECTORY)
        get_filename_component(CEREBRAS_SDK_DIR ${CEREBRAS_SDK_BIN_DIR} DIRECTORY)
    endif()

    # Find include directory
    find_path(CEREBRAS_SDK_INCLUDE_DIR
        NAMES sdk/host/sdk.h
        PATHS ${CEREBRAS_SDK_DIR}
        PATH_SUFFIXES include
    )

    # Find runtime library
    find_library(CEREBRAS_SDK_RUNTIME_LIB
        NAMES cerebras_runtime cs_runtime
        PATHS ${CEREBRAS_SDK_DIR}
        PATH_SUFFIXES lib lib64
    )

    # Get version
    if(CEREBRAS_CSLC)
        execute_process(
            COMMAND ${CEREBRAS_CSLC} --version
            OUTPUT_VARIABLE CSLC_VERSION_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" CEREBRAS_SDK_VERSION "${CSLC_VERSION_OUTPUT}")
    endif()

    # Set results
    if(CEREBRAS_CSLC AND CEREBRAS_SDK_DIR)
        set(CEREBRAS_SDK_FOUND TRUE PARENT_SCOPE)
        set(CEREBRAS_SDK_DIR ${CEREBRAS_SDK_DIR} PARENT_SCOPE)
        set(CEREBRAS_CSLC ${CEREBRAS_CSLC} PARENT_SCOPE)
        set(CEREBRAS_SDK_INCLUDE_DIR ${CEREBRAS_SDK_INCLUDE_DIR} PARENT_SCOPE)
        set(CEREBRAS_SDK_VERSION ${CEREBRAS_SDK_VERSION} PARENT_SCOPE)

        if(CEREBRAS_SDK_RUNTIME_LIB)
            set(CEREBRAS_SDK_LIBRARIES ${CEREBRAS_SDK_RUNTIME_LIB} PARENT_SCOPE)
        endif()

        message(STATUS "Found Cerebras SDK: ${CEREBRAS_SDK_DIR}")
        message(STATUS "  cslc compiler: ${CEREBRAS_CSLC}")
        message(STATUS "  Version: ${CEREBRAS_SDK_VERSION}")
    else()
        set(CEREBRAS_SDK_FOUND FALSE PARENT_SCOPE)
        if(PYFLAME_USE_CEREBRAS_SDK)
            message(WARNING "Cerebras SDK not found. CSL compilation will be disabled.")
            message(WARNING "Set CEREBRAS_SDK_DIR environment variable or install SDK to default location.")
        endif()
    endif()
endfunction()

# Helper function to compile CSL files
function(compile_csl target_name layout_file)
    if(NOT CEREBRAS_SDK_FOUND)
        message(FATAL_ERROR "Cannot compile CSL: Cerebras SDK not found")
    endif()

    set(options "")
    set(oneValueArgs FABRIC_DIMS OUTPUT_DIR)
    set(multiValueArgs EXTRA_FLAGS)
    cmake_parse_arguments(CSL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT CSL_OUTPUT_DIR)
        set(CSL_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/csl_output)
    endif()

    if(NOT CSL_FABRIC_DIMS)
        set(CSL_FABRIC_DIMS "10,10")
    endif()

    # Create output directory
    file(MAKE_DIRECTORY ${CSL_OUTPUT_DIR})

    # Build cslc command
    set(CSLC_COMMAND
        ${CEREBRAS_CSLC}
        ${layout_file}
        --fabric-dims=${CSL_FABRIC_DIMS}
        --fabric-offsets=4,1
        -o ${CSL_OUTPUT_DIR}
        ${CSL_EXTRA_FLAGS}
    )

    add_custom_command(
        OUTPUT ${CSL_OUTPUT_DIR}/out.elf
        COMMAND ${CSLC_COMMAND}
        DEPENDS ${layout_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling CSL: ${layout_file}"
        VERBATIM
    )

    add_custom_target(${target_name}
        DEPENDS ${CSL_OUTPUT_DIR}/out.elf
    )
endfunction()
```

---

## 7. Compiler Options

```cmake
# cmake/CompilerOptions.cmake

function(set_compiler_options)
    # Common flags
    set(PYFLAME_CXX_FLAGS "")

    # Platform-specific flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # GCC and Clang
        list(APPEND PYFLAME_CXX_FLAGS
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wno-unused-parameter
        )

        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND PYFLAME_CXX_FLAGS -g -O0)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            list(APPEND PYFLAME_CXX_FLAGS -O3 -DNDEBUG)
        elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            list(APPEND PYFLAME_CXX_FLAGS -O2 -g -DNDEBUG)
        endif()

        # Sanitizers
        if(PYFLAME_ENABLE_SANITIZERS)
            list(APPEND PYFLAME_CXX_FLAGS
                -fsanitize=address
                -fsanitize=undefined
                -fno-omit-frame-pointer
            )
            link_libraries(-fsanitize=address -fsanitize=undefined)
        endif()

        # Coverage
        if(PYFLAME_ENABLE_COVERAGE)
            list(APPEND PYFLAME_CXX_FLAGS --coverage)
            link_libraries(--coverage)
        endif()

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # Microsoft Visual C++
        list(APPEND PYFLAME_CXX_FLAGS
            /W4
            /WX-
            /permissive-
            /Zc:__cplusplus
        )

        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND PYFLAME_CXX_FLAGS /Od /Zi)
        else()
            list(APPEND PYFLAME_CXX_FLAGS /O2 /DNDEBUG)
        endif()
    endif()

    # Set as global
    add_compile_options(${PYFLAME_CXX_FLAGS})

    # Make available to parent
    set(PYFLAME_CXX_FLAGS ${PYFLAME_CXX_FLAGS} PARENT_SCOPE)
endfunction()
```

---

## 8. Tests CMake

```cmake
# tests/CMakeLists.txt

include(GoogleTest)

# ============================================================================
# C++ unit tests
# ============================================================================

set(CPP_TEST_SOURCES
    cpp/test_tensor.cpp
    cpp/test_dtype.cpp
    cpp/test_layout.cpp
    cpp/test_graph.cpp
    cpp/test_shape_inference.cpp
    cpp/test_passes.cpp
    cpp/test_memory_planner.cpp
    cpp/test_csl_codegen.cpp
)

add_executable(pyflame_tests ${CPP_TEST_SOURCES})

target_link_libraries(pyflame_tests PRIVATE
    pyflame
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(pyflame_tests PRIVATE
    ${PYFLAME_INCLUDE_DIR}
)

# Register tests with CTest
gtest_discover_tests(pyflame_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES LABELS "unit"
)

# ============================================================================
# Integration tests (require Cerebras SDK)
# ============================================================================

if(CEREBRAS_SDK_FOUND)
    set(INTEGRATION_TEST_SOURCES
        cpp/integration/test_simulator.cpp
        cpp/integration/test_elementwise.cpp
        cpp/integration/test_matmul.cpp
    )

    add_executable(pyflame_integration_tests ${INTEGRATION_TEST_SOURCES})

    target_link_libraries(pyflame_integration_tests PRIVATE
        pyflame
        GTest::gtest
        GTest::gtest_main
    )

    target_compile_definitions(pyflame_integration_tests PRIVATE
        PYFLAME_HAS_CEREBRAS_SDK
    )

    gtest_discover_tests(pyflame_integration_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES LABELS "integration"
    )
endif()

# ============================================================================
# Python tests (via pytest)
# ============================================================================

add_custom_target(pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest
        ${CMAKE_CURRENT_SOURCE_DIR}/python
        --tb=short
        -v
    DEPENDS pip-install
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Python tests"
)

# Combined test target
add_custom_target(test-all
    DEPENDS pyflame_tests pytest
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
)
```

---

## 9. pyproject.toml

```toml
# pyproject.toml
[build-system]
requires = [
    "setuptools>=45",
    "wheel",
    "cmake>=3.18",
    "pybind11>=2.10",
]
build-backend = "setuptools.build_meta"

[project]
name = "pyflame"
dynamic = ["version"]
description = "Native deep learning framework for Cerebras WSE"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "PyFlame Team", email = "pyflame@example.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: C++",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["deep learning", "cerebras", "machine learning", "tensor"]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "isort>=5.12",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
docs = [
    "sphinx>=6.0",
    "sphinx-rtd-theme>=1.2",
    "myst-parser>=1.0",
]

[project.urls]
Homepage = "https://github.com/example/pyflame"
Documentation = "https://pyflame.readthedocs.io"
Repository = "https://github.com/example/pyflame"
Issues = "https://github.com/example/pyflame/issues"

[tool.setuptools.dynamic]
version = {attr = "pyflame._version.__version__"}

[tool.setuptools.packages.find]
where = ["python"]

[tool.pytest.ini_options]
testpaths = ["tests/python"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.black]
line-length = 100
target-version = ["py38", "py39", "py310", "py311"]

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

---

## 10. setup.py.in Template

```python
# setup.py.in - Template for generated setup.py
"""
PyFlame: Native deep learning framework for Cerebras WSE
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DPYFLAME_BUILD_TESTS=OFF",
            "-DPYFLAME_BUILD_EXAMPLES=OFF",
        ]

        build_args = ["--config", cfg]

        # Set number of parallel jobs
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["-j", str(os.cpu_count() or 1)]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True
        )


# Read long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="pyflame",
    version="@PROJECT_VERSION@",
    author="PyFlame Team",
    author_email="pyflame@example.com",
    description="Native deep learning framework for Cerebras WSE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/pyflame",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("pyflame._pyflame_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
```

---

## 11. GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-linux:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest numpy pybind11

    - name: Configure CMake
      run: |
        cmake -B build \
          -DPYFLAME_BUILD_PYTHON=ON \
          -DPYFLAME_BUILD_TESTS=ON \
          -DPYFLAME_USE_CEREBRAS_SDK=OFF \
          -DCMAKE_BUILD_TYPE=Release

    - name: Build
      run: cmake --build build -j$(nproc)

    - name: Run C++ tests
      run: ctest --test-dir build --output-on-failure -L unit

    - name: Install Python package
      run: pip install -e build

    - name: Run Python tests
      run: pytest tests/python -v

  build-windows:
    runs-on: windows-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest numpy pybind11

    - name: Configure CMake
      run: |
        cmake -B build `
          -DPYFLAME_BUILD_PYTHON=ON `
          -DPYFLAME_BUILD_TESTS=ON `
          -DPYFLAME_USE_CEREBRAS_SDK=OFF

    - name: Build
      run: cmake --build build --config Release

    - name: Run C++ tests
      run: ctest --test-dir build -C Release --output-on-failure -L unit

  build-macos:
    runs-on: macos-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest numpy pybind11

    - name: Configure CMake
      run: |
        cmake -B build \
          -DPYFLAME_BUILD_PYTHON=ON \
          -DPYFLAME_BUILD_TESTS=ON \
          -DPYFLAME_USE_CEREBRAS_SDK=OFF \
          -DCMAKE_BUILD_TYPE=Release

    - name: Build
      run: cmake --build build -j$(sysctl -n hw.ncpu)

    - name: Run C++ tests
      run: ctest --test-dir build --output-on-failure -L unit

  lint:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Run clang-format
      uses: jidicula/clang-format-action@v4.11.0
      with:
        clang-format-version: '17'
        check-path: 'src'

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Run Python linting
      run: |
        pip install black isort
        black --check python/
        isort --check python/
```

---

## 12. Quick Start Build Instructions

### 12.1 Linux/macOS

```bash
# Clone repository
git clone https://github.com/example/pyflame.git
cd pyflame

# Create build directory
mkdir build && cd build

# Configure (basic)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Configure (with Cerebras SDK)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCEREBRAS_SDK_DIR=/path/to/cerebras/sdk

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure

# Install Python package (development mode)
pip install -e .

# Or system-wide
sudo cmake --install .
```

### 12.2 Windows

```powershell
# Clone repository
git clone https://github.com/example/pyflame.git
cd pyflame

# Create build directory
mkdir build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Run tests
ctest -C Release --output-on-failure

# Install Python package
pip install -e .
```

### 12.3 pip Installation (from source)

```bash
# Install directly from repository
pip install git+https://github.com/example/pyflame.git

# Or from local source
git clone https://github.com/example/pyflame.git
cd pyflame
pip install .

# Development mode
pip install -e ".[dev]"
```

---

## 13. Troubleshooting

### 13.1 Common Issues

| Issue | Solution |
|-------|----------|
| `pybind11 not found` | `pip install pybind11` or install via package manager |
| `Cerebras SDK not found` | Set `CEREBRAS_SDK_DIR` environment variable |
| `C++17 not supported` | Update compiler (GCC 9+, Clang 10+, MSVC 2019+) |
| `Python.h not found` | Install `python3-dev` / `python3-devel` package |
| Build fails on Windows | Use Visual Studio 2019+ with C++ workload |

### 13.2 Debug Build

```bash
# Debug build with sanitizers
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPYFLAME_ENABLE_SANITIZERS=ON

# Debug build with coverage
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPYFLAME_ENABLE_COVERAGE=ON

# Generate coverage report
make coverage
```

---

## 14. Future Enhancements

### 14.1 Planned Build Improvements

1. **Prebuilt Wheels**: Distribute prebuilt wheels for common platforms
2. **Conda Packaging**: Add conda-forge recipe
3. **Docker Images**: Provide development and runtime containers
4. **Cross-Compilation**: Support for ARM and other architectures
5. **Static Analysis**: Integrate clang-tidy and cppcheck

### 14.2 CI/CD Enhancements

1. **Nightly Builds**: Automated builds with latest SDK
2. **Performance Regression**: Track performance across commits
3. **Release Automation**: Automated PyPI/conda releases

---

*Document Version: 1.0*
*Authors: PyFlame Team*
*Last Updated: January 10, 2026*
