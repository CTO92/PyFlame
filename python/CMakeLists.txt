# python/CMakeLists.txt - Python bindings

pybind11_add_module(_pyflame_cpp MODULE
    bindings.cpp
    nn_bindings.cpp
)

target_link_libraries(_pyflame_cpp PRIVATE
    pyflame_core
)

target_include_directories(_pyflame_cpp PRIVATE
    ${PYFLAME_INCLUDE_DIR}
)

# Add ROCm include paths when enabled
if(PYFLAME_USE_ROCM AND DEFINED ROCM_PATH)
    target_include_directories(_pyflame_cpp PRIVATE
        ${ROCM_PATH}/include
    )
    target_compile_definitions(_pyflame_cpp PRIVATE
        PYFLAME_HAS_ROCM
        __HIP_PLATFORM_AMD__
    )
endif()

# Set output directory to match Python package structure
set_target_properties(_pyflame_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pyflame
)

# Copy Python source files
file(GLOB_RECURSE PYTHON_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/pyflame/*.py"
)

foreach(PYTHON_FILE ${PYTHON_SOURCE_FILES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_FILE})
    configure_file(${PYTHON_FILE} ${CMAKE_BINARY_DIR}/python/${REL_PATH} COPYONLY)
endforeach()

# Generate version file
file(WRITE ${CMAKE_BINARY_DIR}/python/pyflame/_version.py
    "__version__ = \"${PROJECT_VERSION}\"\n"
)

# Custom target for pip install in development mode
add_custom_target(pip-install
    COMMAND ${Python3_EXECUTABLE} -m pip install -e ${CMAKE_BINARY_DIR}/python
    DEPENDS _pyflame_cpp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Installing PyFlame in development mode"
)
