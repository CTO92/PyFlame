# src/CMakeLists.txt - PyFlame core library

# Collect source files
set(PYFLAME_CORE_SOURCES
    core/tensor.cpp
    core/tensor_impl.cpp
    core/context.cpp
)

set(PYFLAME_OPS_SOURCES
    ops/elementwise.cpp
    ops/reduction.cpp
    ops/matmul.cpp
)

set(PYFLAME_BACKEND_SOURCES
    backend/csl_codegen.cpp
    backend/csl_emitter.cpp
    backend/executor.cpp
)

# Core library
add_library(pyflame_core STATIC
    ${PYFLAME_CORE_SOURCES}
    ${PYFLAME_OPS_SOURCES}
    ${PYFLAME_BACKEND_SOURCES}
)

target_include_directories(pyflame_core
    PUBLIC
        $<BUILD_INTERFACE:${PYFLAME_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(pyflame_core PUBLIC cxx_std_17)

set_target_properties(pyflame_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME pyflame
)

# Alias for consistent naming
add_library(PyFlame::Core ALIAS pyflame_core)

# Installation
include(GNUInstallDirs)

install(TARGETS pyflame_core
    EXPORT PyFlameTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${PYFLAME_INCLUDE_DIR}/pyflame
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT PyFlameTargets
    FILE PyFlameTargets.cmake
    NAMESPACE PyFlame::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PyFlame
)
