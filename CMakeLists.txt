# PyFlame: Native Deep Learning Framework for Cerebras WSE
# Root CMakeLists.txt
#
# PRE-RELEASE ALPHA 1.0
# This software is in early development and is not yet ready for production use.
# APIs may change without notice.

cmake_minimum_required(VERSION 3.18)

# Version information (Pre-Release Alpha 1.0)
set(PYFLAME_VERSION_MAJOR 1)
set(PYFLAME_VERSION_MINOR 0)
set(PYFLAME_VERSION_PATCH 0)
set(PYFLAME_VERSION_SUFFIX "-alpha")
set(PYFLAME_VERSION_FULL "${PYFLAME_VERSION_MAJOR}.${PYFLAME_VERSION_MINOR}.${PYFLAME_VERSION_PATCH}${PYFLAME_VERSION_SUFFIX}")

project(PyFlame
    VERSION ${PYFLAME_VERSION_MAJOR}.${PYFLAME_VERSION_MINOR}.${PYFLAME_VERSION_PATCH}
    LANGUAGES CXX
    DESCRIPTION "Native deep learning framework for Cerebras WSE (Pre-Release Alpha)"
)

# ============================================================================
# Options
# ============================================================================

option(PYFLAME_BUILD_PYTHON "Build Python bindings" ON)
option(PYFLAME_BUILD_TESTS "Build unit tests" ON)
option(PYFLAME_BUILD_EXAMPLES "Build example programs" ON)
option(PYFLAME_USE_CEREBRAS_SDK "Enable Cerebras SDK integration" OFF)
option(PYFLAME_ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)

# ============================================================================
# CMake configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
# Compiler flags
# ============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
    if(PYFLAME_ENABLE_SANITIZERS)
        add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address -fsanitize=undefined)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi)
    else()
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# ============================================================================
# Dependencies
# ============================================================================

# pybind11 for Python bindings
if(PYFLAME_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # Fetch pybind11 if not found
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
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
    include(GoogleTest)
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
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== PyFlame ${PYFLAME_VERSION_FULL} (Pre-Release Alpha) ===")
message(STATUS "")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "  Build Python:      ${PYFLAME_BUILD_PYTHON}")
message(STATUS "  Build tests:       ${PYFLAME_BUILD_TESTS}")
message(STATUS "  Build examples:    ${PYFLAME_BUILD_EXAMPLES}")
message(STATUS "  Cerebras SDK:      ${PYFLAME_USE_CEREBRAS_SDK}")
message(STATUS "")
message(STATUS "WARNING: This is pre-release alpha software. APIs may change.")
message(STATUS "")
