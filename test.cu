#include <iostream>
#include <vector>
#include "tensor.cuh"
#include <chrono>
#include "cudaCalculate.cuh"

int main() {
    ts::Tensor<double> t=ts::Tensor<double>::rand({1024,1024});
    ts::Tensor<double> t2=ts::Tensor<double>::rand({1024,1024});
    auto start = std::chrono::high_resolution_clock::now();
    cu::cuda_MatrixMul(t, t2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout  << duration.count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    ts::Tensor<double>::matrix_mul(t, t2);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout  << duration.count() << " s" << std::endl;

    return 0;
}
