# examples/CMakeLists.txt - PyFlame example programs

# C++ examples
add_executable(example_basic cpp/basic.cpp)
target_link_libraries(example_basic PRIVATE pyflame_core)
target_include_directories(example_basic PRIVATE ${PYFLAME_INCLUDE_DIR})

add_executable(example_matmul cpp/matmul.cpp)
target_link_libraries(example_matmul PRIVATE pyflame_core)
target_include_directories(example_matmul PRIVATE ${PYFLAME_INCLUDE_DIR})

add_executable(example_mlp cpp/mlp.cpp)
target_link_libraries(example_mlp PRIVATE pyflame_core)
target_include_directories(example_mlp PRIVATE ${PYFLAME_INCLUDE_DIR})
