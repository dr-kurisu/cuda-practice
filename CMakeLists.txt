cmake_minimum_required(VERSION 3.18)

project(cuda-practice LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dst)

add_executable(example
    src/example.cu
)

add_executable(vectorAdd
    src/vectorAdd.cu
)

set_target_properties(example vectorAdd PROPERTIES
    CUDA_ARCHITECTURES 86
)
