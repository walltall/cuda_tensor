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
    __global__ void cuda_mul_kernel(const T* x_data, const T* y_data, T* result_data, size_t rows_x, size_t cols_x, size_t cols_y) {
        size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rows_x && col < cols_y) {
            T sum = 0;
            for (size_t k = 0; k < cols_x; ++k) {
                sum += x_data[row * cols_x + k] * y_data[k * cols_y + col];
            }
            result_data[row * cols_y + col] = sum;
        }
    }
    template<typename T>
    ts::Tensor<T> cuda_mul(const ts::Tensor<T>&input, const ts::Tensor<T>& other) {
        T* x=new T[input.data.size()];
        T* y=new T[other.data.size()];
        for(size_t i=0;i<input.get_size();i++){
            x[i]=*(input.data[i].get());
            y[i]=*(other.data[i].get());
        }
        cudaSetDevice(1);
        size_t size = input.get_size();

        T* result = new T[input.data.size()];

        T* dev_x_data;
        T* dev_y_data;
        T* dev_result_data;
        cudaMalloc((void**)&dev_x_data, size * sizeof(T));
        cudaMalloc((void**)&dev_y_data, size * sizeof(T));
        cudaMalloc((void**)&dev_result_data, size * sizeof(T));

        cudaMemcpy(dev_x_data, x, size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_y_data, y, size * sizeof(T), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;

        cuda_mul_kernel<<<gridSize, blockSize>>>(dev_x_data, dev_y_data, dev_result_data, size);

        cudaDeviceSynchronize();

        cudaMemcpy(result, dev_result_data, size * sizeof(T), cudaMemcpyDeviceToHost);

        cudaFree(dev_x_data);
        cudaFree(dev_y_data);
        cudaFree(dev_result_data);
        ts::Tensor<T>ans=ts::Tensor<T>::zeros(input.get_shape());
        for(size_t i=0;i<ans.get_size();i++){
            *(ans.data[i])=result[i];
        }
        free(result);
        free(x);
        free(y);
        return ans;
    }

    template<typename T>
    ts::Tensor<T> cuda_MatrixMul(const ts::Tensor<T>& input, const ts::Tensor<T>& other) {
        cudaSetDevice(1);
        size_t rows_x = input.get_shape()[0];
        size_t cols_x = input.get_shape()[1];
        size_t rows_y = other.get_shape()[0];
        size_t cols_y = other.get_shape()[1];
        T* x=new T[input.data.size()];
        T* y=new T[other.data.size()];
        for(size_t i=0;i<input.get_size();i++){
            x[i]=*(input.data[i].get());
        }
        for(size_t i=0;i<other.get_size();i++){
            y[i]=*(other.data[i].get());
        }
        T* result=new T[rows_x*cols_y];

        T* dev_x_data;
        T* dev_y_data;
        T* dev_result_data;
        cudaMalloc((void**)&dev_x_data, rows_x * cols_x * sizeof(T));
        cudaMalloc((void**)&dev_y_data, rows_y * cols_y * sizeof(T));
        cudaMalloc((void**)&dev_result_data, rows_x * cols_y * sizeof(T));

        cudaMemcpy(dev_x_data, x, rows_x * cols_x * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_y_data, y, rows_y * cols_y * sizeof(T), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((cols_y + blockSize.x - 1) / blockSize.x, (rows_x + blockSize.y - 1) / blockSize.y);

        cuda_mul_kernel<<<gridSize, blockSize>>>(dev_x_data, dev_y_data, dev_result_data, rows_x, cols_x, cols_y);

        cudaDeviceSynchronize();

        cudaMemcpy(result, dev_result_data, rows_x * cols_y * sizeof(T), cudaMemcpyDeviceToHost);

        cudaFree(dev_x_data);
        cudaFree(dev_y_data);
        cudaFree(dev_result_data);
        ts::Tensor<T> ans = ts::Tensor<T>::zeros({rows_x, cols_y});
        for(size_t i=0;i<ans.get_size();i++){
            *(ans.data[i])=result[i];
        }
        free(result);
        free(x);
        free(y);
        return ans;
    }

}  // namespace cu