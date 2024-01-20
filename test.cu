#include <iostream>
#include <vector>
#include "tensor.cuh"
#include <chrono>
#include "cudaCalculate.cuh"

int main() {
    ts::Tensor<double> t=ts::Tensor<double>::rand({1024,1024});
    ts::Tensor<double> t2=ts::Tensor<double>::rand({1024,1024});
//    std::cout<<t<<std::endl;
//    std::cout<<t2<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    ts::Tensor<double>res=cu::cuda_MatrixMul(t, t2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout  << duration.count() << " s" << std::endl;
//    std::cout<<res<<std::endl;

    start = std::chrono::high_resolution_clock::now();
    res=ts::Tensor<double>::matrix_mul(t, t2);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout  << duration.count() << " s" << std::endl;
//    std::cout<<res<<std::endl;


    return 0;
}
