# tests/CMakeLists.txt - PyFlame unit tests

include(GoogleTest)

# C++ unit tests
add_executable(pyflame_tests
    # Phase 1 tests
    cpp/test_dtype.cpp
    cpp/test_layout.cpp
    cpp/test_tensor.cpp
    cpp/test_graph.cpp
    cpp/test_shape_inference.cpp
    cpp/test_operations.cpp
    # Phase 2 tests
    cpp/test_autograd.cpp
    cpp/test_nn_layers.cpp
    cpp/test_loss.cpp
    cpp/test_optim.cpp
)

target_link_libraries(pyflame_tests PRIVATE
    pyflame_core
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

# ROCm backend tests (conditional)
if(PYFLAME_USE_ROCM)
    add_executable(pyflame_rocm_tests
        backend/rocm/test_rocm_device.cpp
        backend/rocm/test_rocm_memory.cpp
        backend/rocm/test_rocm_kernels.cpp
        backend/rocm/test_rocm_operations.cpp
    )

    target_link_libraries(pyflame_rocm_tests PRIVATE
        pyflame_core
        GTest::gtest
        GTest::gtest_main
        hip::device
        roc::rocblas
        MIOpen
    )

    target_include_directories(pyflame_rocm_tests PRIVATE
        ${PYFLAME_INCLUDE_DIR}
        ${ROCM_PATH}/include
    )

    target_compile_definitions(pyflame_rocm_tests PRIVATE
        PYFLAME_HAS_ROCM
    )

    # Register ROCm tests with CTest
    gtest_discover_tests(pyflame_rocm_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES LABELS "rocm"
    )
endif()

# Python tests (via pytest)
if(PYFLAME_BUILD_PYTHON)
    add_custom_target(pytest
        COMMAND ${Python3_EXECUTABLE} -m pytest
            ${CMAKE_CURRENT_SOURCE_DIR}/python
            --tb=short
            -v
        DEPENDS _pyflame_cpp
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running Python tests"
    )

    # ROCm-specific Python tests
    if(PYFLAME_USE_ROCM)
        add_custom_target(pytest_rocm
            COMMAND ${Python3_EXECUTABLE} -m pytest
                ${CMAKE_CURRENT_SOURCE_DIR}/python/test_rocm_backend.py
                --tb=short
                -v
            DEPENDS _pyflame_cpp
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running ROCm Python tests"
        )
    endif()
endif()
