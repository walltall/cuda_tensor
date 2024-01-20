// cudaCalculate.cuh

#pragma once
#include "tensor.cuh"
#include <cuda_runtime.h>

namespace cu {
    template<typename T>
    __global__ void cuda_mul_kernel(const T* x_data, const T* y_data, T* result_data, size_t size) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
            result_data[tid] = x_data[tid] * y_data[tid];
        }
    }

    template<typename T>
    ts::Tensor<T> cuda_mul(const ts::Tensor<T>& x, const ts::Tensor<T>& y) {
        cudaSetDevice(1);
        size_t size = x.get_size();

        ts::Tensor<T> result = ts::Tensor<T>::zeros(x.get_shape());

        T* dev_x_data;
        T* dev_y_data;
        T* dev_result_data;
        cudaMalloc((void**)&dev_x_data, size * sizeof(T));
        cudaMalloc((void**)&dev_y_data, size * sizeof(T));
        cudaMalloc((void**)&dev_result_data, size * sizeof(T));

        cudaMemcpy(dev_x_data, x.data.get(), size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_y_data, y.data.get(), size * sizeof(T), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;

        cuda_mul_kernel<<<gridSize, blockSize>>>(dev_x_data, dev_y_data, dev_result_data, size);

        cudaDeviceSynchronize();

        cudaMemcpy(result.data.get(), dev_result_data, size * sizeof(T), cudaMemcpyDeviceToHost);

        cudaFree(dev_x_data);
        cudaFree(dev_y_data);
        cudaFree(dev_result_data);

        return result;
    }

}  // namespace cu