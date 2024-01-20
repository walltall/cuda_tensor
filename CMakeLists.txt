cmake_minimum_required(VERSION 3.27)
project(cuda_project CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_project
        tensor.cuh test.cu
cudaCalculate.cuh)

set_target_properties(cuda_project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
