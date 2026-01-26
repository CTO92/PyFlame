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

# Phase 2: Autograd sources
set(PYFLAME_AUTOGRAD_SOURCES
    autograd/grad_mode.cpp
    autograd/autograd.cpp
)

# Phase 2: Neural network sources
set(PYFLAME_NN_SOURCES
    nn/module.cpp
    nn/linear.cpp
    nn/conv.cpp
    nn/normalization.cpp
    nn/pooling.cpp
    nn/dropout.cpp
    nn/attention.cpp
    nn/loss.cpp
)

# Phase 2: Optimizer sources
set(PYFLAME_OPTIM_SOURCES
    optim/optimizer.cpp
    optim/lr_scheduler.cpp
)

# ROCm backend sources (conditional)
if(PYFLAME_USE_ROCM)
    # C++ sources for ROCm backend
    set(PYFLAME_ROCM_SOURCES
        backend/rocm/rocm_device.cpp
        backend/rocm/rocm_memory.cpp
        backend/rocm/rocm_blas.cpp
        backend/rocm/rocm_dnn.cpp
        backend/rocm/rocm_executor.cpp
        backend/rocm/rocm_perf.cpp
    )

    # HIP kernel sources (compiled with HIP compiler)
    set(PYFLAME_HIP_SOURCES
        backend/rocm/rocm_kernels.hip
    )
endif()

# Core library
add_library(pyflame_core STATIC
    ${PYFLAME_CORE_SOURCES}
    ${PYFLAME_OPS_SOURCES}
    ${PYFLAME_BACKEND_SOURCES}
    ${PYFLAME_AUTOGRAD_SOURCES}
    ${PYFLAME_NN_SOURCES}
    ${PYFLAME_OPTIM_SOURCES}
    ${PYFLAME_ROCM_SOURCES}  # Will be empty if ROCm disabled
    ${PYFLAME_HIP_SOURCES}   # HIP kernels (will be empty if ROCm disabled)
)

# Set HIP source file properties (language detection)
if(PYFLAME_USE_ROCM)
    set_source_files_properties(${PYFLAME_HIP_SOURCES}
        PROPERTIES LANGUAGE HIP
    )
endif()

target_include_directories(pyflame_core
    PUBLIC
        $<BUILD_INTERFACE:${PYFLAME_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(pyflame_core PUBLIC cxx_std_17)

# ROCm library linking (conditional)
if(PYFLAME_USE_ROCM)
    target_link_libraries(pyflame_core
        PRIVATE
        hip::device
        roc::rocblas
        MIOpen
    )

    target_include_directories(pyflame_core
        PRIVATE
        ${ROCM_PATH}/include
    )
endif()

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
