#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <omp.h>
namespace ts {
    template <typename T>
    class Tensor;

    template <typename TT>
    struct broadcast_tensor{
        std::shared_ptr<Tensor<TT>> tensor1;
        std::shared_ptr<Tensor<TT>> tensor2;
        broadcast_tensor(const Tensor<TT>& t1, const Tensor<TT>& t2) : tensor1(std::make_shared<Tensor<TT>>(t1)), tensor2(std::make_shared<Tensor<TT>>(t2)) {}
    };

    class IndexError : public std::exception {
    private:
        std::string message;
    public:
        IndexError(std::string message){
            this->message = message;
        }

        const char* what() const noexcept override {
                return message.c_str();
        }
    };

    class BroadcastError : public std::exception {
    private:
        std::string message;
    public:
        BroadcastError(std::string message){
            this->message = message;
        }

        const char* what() const noexcept override {
                return message.c_str();
        }
    };

    class EmptyVectorError : public std::exception {
    private:
        std::string message;
    public:
        EmptyVectorError(std::string message){
            this->message = message;
        }

        const char* what() const noexcept override {
                return message.c_str();
        }
    };

    class DimensionError : public std::exception {
    private:
        std::string message;
    public:
        DimensionError(std::string message){
            this->message = message;
        }

        const char* what() const noexcept override {
                return message.c_str();
        }
    };

    class ShapeError : public std::exception {
    private:
        std::string message;
    public:
        ShapeError(std::string message){
            this->message = message;
        }

        const char* what() const noexcept override {
                return message.c_str();
        }
    };

    class ExpressionMismatchError : public std::exception {
    private:
        std::string message;
    public:
        ExpressionMismatchError(std::string message){
            this->message = message;
        }

        const char* what() const noexcept override {
                return message.c_str();
        }
    };

    class ExpressionNotSupportedError : public std::exception {
    private:
        std::string message;
    public:
        ExpressionNotSupportedError(std::string message){
            this->message = message;
        }

        const char* what() const noexcept override {
                return message.c_str();
        }
    };

    class OpenFileError : public std::exception {
    private:
        std::string message;
    public:
        OpenFileError(std::string message){
            this->message = message;
        }

        const char* what() const noexcept override {
                return message.c_str();
        }
    };

    template <typename T>
    class Tensor {
    private:
        std::vector<size_t> shape;
        std::vector<size_t> strides;
        size_t dims;
        size_t size;

        template<typename U, size_t M>
        int init(U (&data)[M], size_t offset){
            size_t len = M;
            if(offset == 0) {
                this->shape.push_back(len);
                this->size *= len;
            }
            int stride = sizeof(data[0]);
            int s = 1;
            for(size_t i = 0; i < len; i++){
                int tmp = init(data[i], offset);
                if(offset == 0) {
                    s *= tmp;
                    this->strides.push_back(s);
                }
                offset += stride;
            }
            return s * len;
        }

        int init(T data, size_t offset){
            if(offset == 0){
                this->data.resize(this->size);
                for (size_t i = 0; i < this->size; ++i) {
                    this->data[i] = std::make_shared<T>(T());
                }
            }
            *(this->data[offset / sizeof(T)]) = data;
            return 1;
        }

        void randInt(const std::vector<size_t>& dimensions) {
            size_t totalSize = 1;
            for (size_t dim : dimensions) {
                totalSize *= dim;
            }
            for (size_t i = 0; i < totalSize; ++i) {
                this->data[i] = std::make_shared<T>(T());
                *(this->data[i].get()) = std::rand() % 100;
            }
        }

        // Helper function for rand that generates random floats
        void randFloat(const std::vector<size_t>& dimensions) {
            size_t totalSize = 1;
            for (size_t dim : dimensions) {
                totalSize *= dim;
            }

            for (size_t i = 0; i < totalSize; ++i) {
                float r = 0 + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(100 - 0)));
                this->data[i] = std::make_shared<T>(T());
                *(this->data[i].get()) = r;
            }
        }

    public:
        std::vector<std::shared_ptr<T>> data;
        Tensor() = default;
        std::vector<size_t> get_shape()const{
            return shape;
        }
        std::size_t get_size()const{
            return size;
        }

        template<typename U, size_t M>
        Tensor(U (&data)[M]) {
            size_t len = M;
            this->shape.push_back(len);
            this->size = len;
            size_t offset = 0;
            size_t stride = sizeof(data[0]);
            int s = 1;
            for(size_t i = 0; i < len; i++) {
                s *= init(data[i], offset);
                if(i == 0) this->strides.push_back(s);
                offset += stride;
            }
            this->dims = this->shape.size();
            std::reverse(this->strides.begin(),this->strides.end());
        }

        Tensor(std::vector<std::shared_ptr<T>> data, size_t size, const std::vector<size_t>& dimensions) {
            this->data.resize(size);
            for(int i = 0; i < size; i++){
                this->data[i] = data[i];
            }
            this->size = size;
            this->shape = dimensions;

            this->dims = dimensions.size();
            this->strides.resize(this->dims);

            for (size_t i = 0; i < this->dims; ++i) {
                size /= dimensions[i];
                this->strides[i] = size;
            }
        }

        Tensor(const Tensor<T> & other){
            this->shape = other.shape;
            this->strides = other.strides;
            this->size = other.size;
            this->dims = other.dims;
            this->data.resize(this->size);
            for(int i = 0; i < this->size; i++) {
                this->data[i] = std::make_shared<T>(T());
                *(this->data[i].get()) = *(other.data[i].get());
            }
        }

        static Tensor<T> rand(const std::vector<size_t>& dimensions) {
            Tensor<T> randomTensor;
            randomTensor.shape = dimensions;

            size_t totalSize = 1;
            for (size_t dim : randomTensor.shape) {
                totalSize *= dim;
            }
            randomTensor.size = totalSize;
            randomTensor.data.resize(totalSize);
            if (std::is_integral<T>::value) {
                randomTensor.randInt(dimensions);
            } else if (std::is_floating_point<T>::value) {
                randomTensor.randFloat(dimensions);
            }

            size_t stride = sizeof(T);
            randomTensor.dims = dimensions.size();
            randomTensor.strides.resize(randomTensor.dims);

            for (size_t i = 0; i < randomTensor.dims; ++i) {
                totalSize /= dimensions[i];
                randomTensor.strides[i] = totalSize;
            }

            return randomTensor;
        }

        static Tensor<T> zeros(const std::vector<size_t>& dimensions) {
            Tensor<T> zeroTensor;
            zeroTensor.shape = dimensions;

            size_t totalSize = 1;
            for (size_t dim : zeroTensor.shape) {
                totalSize *= dim;
            }
            zeroTensor.size = totalSize;
            zeroTensor.data.resize(totalSize);
            for(int i = 0; i < totalSize; i++) {
                zeroTensor.data[i] = std::make_shared<T>(T());
                *(zeroTensor.data[i].get()) = 0;
            }

            size_t stride = sizeof(T);
            zeroTensor.dims = dimensions.size();
            zeroTensor.strides.resize(zeroTensor.dims);

            for (size_t i = 0; i < zeroTensor.dims; ++i) {
                totalSize /= dimensions[i];
                zeroTensor.strides[i] = totalSize;
            }

            return zeroTensor;
        }

        static Tensor<T> ones(const std::vector<size_t>& dimensions) {
            Tensor<T> oneTensor;
            oneTensor.shape = dimensions;

            size_t totalSize = 1;
            for (size_t dim : oneTensor.shape) {
                totalSize *= dim;
            }
            oneTensor.size = totalSize;


            oneTensor.data.resize(totalSize);
            for(int i = 0; i < totalSize; i++) {
                oneTensor.data[i] = std::make_shared<T>(T());
                *(oneTensor.data[i].get()) = 1;
            }

            size_t stride = sizeof(T);
            oneTensor.dims = dimensions.size();
            oneTensor.strides.resize(oneTensor.dims);

            for (size_t i = 0; i < oneTensor.dims; ++i) {
                totalSize /= dimensions[i];
                oneTensor.strides[i] = totalSize;
            }

            return oneTensor;
        }

        static Tensor<T> full(const std::vector<size_t>& dimensions, T value) {
            Tensor<T> fullTensor;
            fullTensor.shape = dimensions;

            size_t totalSize = 1;
            for (size_t dim : fullTensor.shape) {
                totalSize *= dim;
            }
            fullTensor.size = totalSize;
            fullTensor.data.resize(totalSize);
            for(int i = 0; i < totalSize; i++) {
                fullTensor.data[i] = std::make_shared<T>(T());
                *(fullTensor.data[i].get()) = value;
            }

            size_t stride = sizeof(T);
            fullTensor.dims = dimensions.size();
            fullTensor.strides.resize(fullTensor.dims);

            for (size_t i = 0; i < fullTensor.dims; ++i) {
                totalSize /= dimensions[i];
                fullTensor.strides[i] = totalSize;
            }

            return fullTensor;
        }

        static Tensor<T> eye(int row) {
            Tensor<T> eyeTensor;
            eyeTensor.shape.push_back(row);
            eyeTensor.shape.push_back(row);

            size_t totalSize = 1;
            for (size_t dim : eyeTensor.shape) {
                totalSize *= dim;
            }
            eyeTensor.size = totalSize;
            eyeTensor.data.resize(totalSize);
            for (size_t i = 0; i < totalSize; ++i) {
                if(i / row == i % row) {
                    eyeTensor.data[i] = std::make_shared<T>(T());
                    *(eyeTensor.data[i].get()) = 1;
                } else {
                    eyeTensor.data[i] = std::make_shared<T>(T());
                    *(eyeTensor.data[i].get()) = 0;
                }
            }

            size_t stride = sizeof(T);
            eyeTensor.dims = 2;
            eyeTensor.strides.resize(eyeTensor.dims);

            for (size_t i = 0; i < eyeTensor.dims; ++i) {
                totalSize /= eyeTensor.shape[i];
                eyeTensor.strides[i] = totalSize;
            }

            return eyeTensor;
        }

        static Tensor<T> empty(const std::vector<size_t>& dimensions) {
            Tensor<T> emptyTensor;
            emptyTensor.shape = dimensions;

            size_t totalSize = 1;
            for (size_t dim : emptyTensor.shape) {
                totalSize *= dim;
            }
            emptyTensor.size = totalSize;
            emptyTensor.data.resize(totalSize);
            for(int i = 0; i < totalSize; i++) {
                emptyTensor.data[i] = std::make_shared<T>(T());
            }

            size_t stride = sizeof(T);
            emptyTensor.dims = dimensions.size();
            emptyTensor.strides.resize(emptyTensor.dims);

            for (size_t i = 0; i < emptyTensor.dims; ++i) {
                totalSize /= dimensions[i];
                emptyTensor.strides[i] = totalSize;
            }
            return emptyTensor;
        }

        static Tensor<T> logspace(double start, double end, int steps, double base = 10){
            if(typeid(T) != typeid(double)) throw ExpressionNotSupportedError("ExpressionNotSupportedError: The function can only create double tensor");
            Tensor<T> tensor = empty({(long long unsigned int)steps});
            double stride = (end - start) / (steps - 1);
            for(int i = 0; i < steps; i++){
                double cur = pow(base, start + i * stride);
                *(tensor.data[i].get()) = cur;
            }
            return tensor;
        }


        Tensor<T> operator()(size_t index) {
            if(index >= shape[0] || index < 0) {
                throw IndexError("IndexError: Index is out of bound");
            }
            std::vector<size_t> dimensions;
            for(int i = 1; i < this->dims; i++) dimensions.push_back(this->shape[i]);
            if(dimensions.empty()) dimensions.push_back(1);
            int indexSize = this->size / shape[0];
            std::vector<std::shared_ptr<T>> indexData;
            int base = index * this->strides[0];
            for(int i = 0; i < strides[0]; i++) {
                std::shared_ptr<T> ptr = this->data[base + i];
                indexData.push_back(ptr);
            }
            Tensor<T> indexT(indexData, indexSize, dimensions);
            return indexT;
        }

        Tensor<T> operator()(size_t location, const std::vector<size_t>& indices) {
            if((location >= this->shape[0] || location < 0) || (indices[0] >= this->shape[0] || indices[0] < 0) || (indices[1] >= this->shape[0] || indices[1] < 0)) {

            }
            int sliceSize = indices[1] - indices[0];
            int startIndex = indices[0];
            std::vector<size_t> dimensions;
            dimensions.push_back(sliceSize);
            for(int i = 2; i < dims; i++){
                dimensions.push_back(this->shape[i]);
            }
            sliceSize *= this->strides[1];
            std::vector<std::shared_ptr<T>> sliceData;
            int base = location * this->strides[0] + startIndex * this->strides[1];
            for(int i = 0; i < strides[0]; i++) {
                std::shared_ptr<T> ptr = this->data[base + i];
                sliceData.push_back(ptr);
            }
            Tensor<T> spliceT(sliceData, sliceSize, dimensions);
            return spliceT;
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
            os << "[";
            printTensor(os, tensor, 0,0);
            os << "]";
            return os;
        }


        //-------------Tensor_part2:
        //-------------MATH:
        //---add
//        template <typename U>
//        ts::Tensor<T>& add(const ts::Tensor<U>& other) const {
//            ts::Tensor<T> result = zeros(other.shape);
//            for (size_t i = 0; i < result.size; ++i) {
//                result.data[i] = this->data[i] + static_cast<T>(other.data[i]);
//            }
//            return *result;
//        }


        static broadcast_tensor<T> broadcast(const Tensor<T> & x,const Tensor<T> & y){
            //对比x,y的维度是否符合广播条件
            int x_end = x.shape.size() - 1;
            int y_end = y.shape.size() - 1;
            std::vector<int> x_broad;
            std::vector<int> y_broad;
            while(x_end >= 0 && y_end >= 0){
                if(x.shape[x_end] == y.shape[y_end]){
                    x_broad.insert(x_broad.begin(), 1);
                    y_broad.insert(y_broad.begin(), 1);
                } else if(x.shape[x_end] == 1){
                    x_broad.insert(x_broad.begin(), (int)y.shape[y_end]);
                    y_broad.insert(y_broad.begin(), 1);
                } else if(y.shape[y_end] == 1){
                    x_broad.insert(x_broad.begin(), 1);
                    y_broad.insert(y_broad.begin(), (int)x.shape[x_end]);
                } else{
                    throw BroadcastError("BroadcastError: The input is not broadcastable");
                }
                x_end--;
                y_end--;
            }
            while (x_end >= 0){
                x_broad.insert(x_broad.begin(), 1);
                y_broad.insert(y_broad.begin(), (int)x.shape[x_end]);
                x_end--;
            }
            while (y_end >= 0){
                x_broad.insert(x_broad.begin(), (int)y.shape[y_end]);
                y_broad.insert(y_broad.begin(), 1);
                y_end--;
            }
            Tensor<T> broadcast1 = Tensor<T>::tile(x, x_broad);
            Tensor<T> broadcast2 = Tensor<T>::tile(y, y_broad);
            broadcast_tensor<T> broadcast_t(broadcast1, broadcast2);
            return broadcast_t;
            //检查完毕，开始广播：
//            std::vector<size_t> broadcast_Shape;
//            int max_dim = std::max(x.shape.size(),y.shape.size());
//            broadcast_Shape.resize(max_dim);
//            int x_idx = x.shape.size()-1;
//            int y_idx = y.shape.size()-1;
//            int broadcast_idx = max_dim -1;
//            while(broadcast_idx >=0){
//                size_t dim_x = (x_idx>=0) ? x.shape[x_idx] : 1;
//                size_t dim_y = (y_idx>=0) ? y.shape[y_idx] : 1;
//                broadcast_Shape[broadcast_idx] = std::max(dim_x,dim_y);
//                x_idx--;
//                y_idx--;
//                broadcast_idx--;
//            }
//            return broadcast_Shape;
        }

        static Tensor<T> broadcasting(const Tensor<T>& x, const std::vector<size_t>& broadcastShape) {
            Tensor<T> result = zeros(broadcastShape);
            // 获取 x 张量的形状和维度
            const std::vector<size_t>& xShape = x.shape;
            size_t xDims = x.dims;

            // 计算广播规则下的 strides
            std::vector<size_t> strides(xDims);
            size_t stride = 1;
            for (int i = xDims - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= xShape[i];
            }
            // 使用循环遍历 result 张量的所有元素，并根据广播规则赋值
            std::vector<size_t> xIndices(xDims, 0);  // 用于保存当前元素在 x 张量中的索引
            for (size_t i = 0; i < result.size; ++i) {
                // 将对应位置的 x 元素赋值给 result
                T value = *(x.data[0].get());  // 默认使用 x 张量中的第一个元素
                for (int j = 0; j < xDims; ++j) {
                    if (xShape[j] > 1) {
                        value = *(x.data[xIndices[j]].get());  // 使用对应位置的 x 元素
                        break;
                    }
                }
                *(result.data[i].get()) = value;

                // 更新 xIndices，以便获取下一个元素
                for (int j = xDims - 1; j >= 0; --j) {
                    if (xShape[j] == 1) {
                        continue;  // 跳过维度为 1 的轴
                    }
                    xIndices[j] += 1;
                    if (xIndices[j] == xShape[j]) {
                        xIndices[j] = 0;  // 循环访问 x 张量中的元素
                    } else {
                        break;
                    }
                }
            }
            return result;
        }

        // 2.2 joining
        // concat
        static Tensor<T> cat(std::vector<Tensor<T>> tensors, int dim){
            if(tensors.size() == 0){
                throw EmptyVectorError("EmptyVectorError: Please input at least 1 tensor");
            }
            if(dim < 0){
                throw DimensionError("DimensionError: Cannot cat in negative dimension");
            }
            Tensor<T> mainTensor = tensors[0];
            size_t mainDim = mainTensor.shape[dim];
            std::vector<size_t> newDimension = mainTensor.shape;
            for(size_t i = 1; i < tensors.size(); i++){
                newDimension[dim] += tensors[i].shape[dim];
            }
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++){
                std::shared_ptr<T> dst_ptr = trans.data[i];
                size_t k = 0;
                size_t src_index = 0;
                size_t dst_index = i;
                std::vector<size_t> coordinates;
                for (size_t j = 0; j < trans.strides.size(); ++j) {
                    size_t coordinate = dst_index / trans.strides[j];
                    dst_index %= trans.strides[j];
                    if(j == dim){
                        size_t src_coordinate = coordinate;
                        while (src_coordinate >= tensors[k].shape[dim]){
                            src_coordinate -= tensors[k].shape[dim];
                            k++;
                        }
                        coordinates.push_back(src_coordinate);
                    } else{
                        coordinates.push_back(coordinate);
                    }
                }
                for(size_t j = 0; j < coordinates.size(); j++){
                    src_index += coordinates[j] * tensors[k].strides[j];
                }
                std::shared_ptr<T> src_ptr = tensors[k].data[src_index];
                *dst_ptr = *src_ptr;
            }
//            std::cout << trans << std::endl;
            return trans;
        }

        // tile
        static Tensor<T> tile(Tensor<T> tensor, std::vector<int> dims){
            std::vector<size_t> newDimension = tensor.shape;
            if(tensor.dims > dims.size()){
                dims.insert(dims.begin(), tensor.dims - dims.size(), 1);
            }else if(tensor.dims < dims.size()) {
                newDimension.insert(newDimension.begin(), dims.size() - tensor.dims, 1);
            }
            Tensor<T> tensor_copy = Tensor<T>::zeros(newDimension);
            tensor_copy.data = tensor.data;
            for(size_t i = 0; i < dims.size(); i++){
                if(dims[i] <= 0){
                    //error
                    throw DimensionError("DimensionError: Cannot tile with non-positive dimensions");
                } else{
                    newDimension[i] *= dims[i];
                }
            }
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for (size_t i = 0; i < trans.size; ++i) {
                std::shared_ptr<T> dst_ptr = trans.data[i];
                size_t src_index = 0;
                size_t dst_index = i;
                for (size_t j = 0; j < trans.strides.size(); ++j) {
                    size_t coordinate = dst_index / trans.strides[j];
                    dst_index %= trans.strides[j];
                    coordinate %= tensor_copy.shape[j];
                    src_index += coordinate * tensor_copy.strides[j];
                }
                std::shared_ptr<T> src_ptr = tensor_copy.data[src_index];
                *dst_ptr = *src_ptr;
            }
            return trans;
        }

        // 2.3 mutation
        // used for scalar assignment to part of a tensor
        // param:
        //     this: the pointer to indexed tensor(processed)
        //     scalar: the value for assignment
        // maybe can return scalar to chain assign
        void operator=(T scalar) {
            size_t length = this->shape.front();
            for (size_t i = 0; i < length; i++) {
                std::shared_ptr<T> data_ptr = this->data[i];
                *data_ptr = scalar;
            }
        }

        // used for tensor assignment to part of a tensor
        void operator=(Tensor<T> operand){
            size_t length = operand.shape.front();
            for(size_t i = 0; i < length; i++){
                *(this->data[i]) = *(operand.data[i]);
            }
        }

        // 2.4 transpose & permutation
        //transpose
        static Tensor<T> transpose(Tensor<T> & tensor, int dim1, int dim2) {
            if(dim1 < 0 || dim2 < 0){
                throw DimensionError("DimensionError: Cannot transpose on negative dimensions");
            }
            std::vector<size_t> newDimension = tensor.shape;
            newDimension[dim1] = tensor.shape[dim2];
            newDimension[dim2] = tensor.shape[dim1];
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            std::vector<std::shared_ptr<T>> temp_ptr = tensor.data;
            for(size_t i = 0; i < trans.size; i++) {
                size_t src_index = i;
                size_t dst_index = 0;
                for (size_t j = 0; j < tensor.strides.size(); ++j) {
                    size_t coordinate = src_index / tensor.strides[j];
                    src_index %= tensor.strides[j];
                    if(j == dim1){
                        dst_index += coordinate * trans.strides[dim2];
                    } else if(j == dim2){
                        dst_index += coordinate * trans.strides[dim1];
                    } else{
                        dst_index += coordinate * trans.strides[j];
                    }
                }
                trans.data[dst_index] = tensor.data[i];
            }
            return trans;
        }

        Tensor<T> transpose(int dim1, int dim2) {
            if(dim1 < 0 || dim2 < 0){
                throw DimensionError("DimensionError: Cannot transpose on negative dimensions");
            }
            std::vector<size_t> newDimension = this->shape;
            newDimension[dim1] = this->shape[dim2];
            newDimension[dim2] = this->shape[dim1];
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++) {
                size_t src_index = i;
                size_t dst_index = 0;
                for (size_t j = 0; j < this->strides.size(); ++j) {
                    size_t coordinate = src_index / this->strides[j];
                    src_index %= this->strides[j];
                    if(j == dim1){
                        dst_index += coordinate * trans.strides[dim2];
                    } else if(j == dim2){
                        dst_index += coordinate * trans.strides[dim1];
                    } else{
                        dst_index += coordinate * trans.strides[j];
                    }
                }
                trans.data[dst_index] = this->data[i];
            }
            return trans;
        }

        // permutation
        static Tensor<T> permute(Tensor<T> & tensor, const std::vector<size_t> & dim){
            std::vector<size_t> newDimension = tensor.shape;
            for(size_t i = 0; i < dim.size(); i++) {
                if(dim[i] < 0){
                    throw DimensionError("DimensionError: Cannot permute on negative dimensions");
                }
                newDimension[i] = tensor.shape[dim[i]];
            }
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++) {
                size_t src_index = 0;
                size_t dst_index = i;
                for (size_t j = 0; j < trans.strides.size(); ++j) {
                    size_t coordinate = dst_index / trans.strides[j];
                    dst_index %= trans.strides[j];
                    src_index += coordinate * tensor.strides[dim[j]];
                }
                trans.data[i] = tensor.data[src_index];
            }
            return trans;
        }

        Tensor<T> permute(const std::vector<size_t> & dim){
            std::vector<size_t> newDimension = this->shape;
            for(size_t i = 0; i < dim.size(); i++) {
                if(dim[i] < 0){
                    throw DimensionError("DimensionError: Cannot permute on negative dimensions");
                }
                newDimension[i] = this->shape[dim[i]];
            }
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++) {
                size_t src_index = 0;
                size_t dst_index = i;
                for (size_t j = 0; j < trans.strides.size(); ++j) {
                    size_t coordinate = dst_index / trans.strides[j];
                    dst_index %= trans.strides[j];
                    src_index += coordinate * this->strides[dim[j]];
                }
                trans.data[i] = this->data[src_index];
            }
            return trans;
        }

        // 2.5 view
        static Tensor<T> view(Tensor<T> & tensor, const std::vector<int> shape){
            std::vector<int> newDimension = shape;
            size_t full_size = 1;
            int reserve = -1;
            for(size_t i = 0; i < shape.size(); i++){
                int dim = shape[i];
                if(dim > 0){
                    full_size *= dim;
                } else if(dim == -1 && reserve == -1){
                    reserve = i;
                } else{
                    // error
                    throw DimensionError("DimensionError: Invalid dimensions");
                }
            }
            if(reserve != -1){
                size_t lost = tensor.size / full_size;
                newDimension[reserve] = lost;
                full_size *= lost;
            }
            if(tensor.size != full_size) {
                //error
                throw ShapeError("ShapeError: Shape does not match");
            }
            std::vector<size_t> dimensions;
            for(size_t i = 0; i < newDimension.size(); i++) dimensions.push_back(newDimension[i]);
            Tensor<T> trans = Tensor<T>::zeros(dimensions);
            trans.data = tensor.data;
            return trans;
        }

        Tensor<T> view(const std::vector<int> shape){
            std::vector<int> newDimension = shape;
            size_t full_size = 1;
            int reserve = -1;
            for(size_t i = 0; i < shape.size(); i++){
                int dim = shape[i];
                if(dim > 0){
                    full_size *= dim;
                } else if(dim == -1 && reserve == -1){
                    reserve = i;
                } else{
                    // error
                    throw DimensionError("DimensionError: Invalid dimensions");
                }
            }
            if(reserve != -1){
                size_t lost = this->size / full_size;
                newDimension[reserve] = lost;
                full_size *= lost;
            }
            if(this->size != full_size) {
                //error
                throw ShapeError("ShapeError: Shape does not match");
            }
            std::vector<size_t> dimensions;
            for(size_t i = 0; i < newDimension.size(); i++) dimensions.push_back(newDimension[i]);
            Tensor<T> trans = Tensor<T>::zeros(dimensions);
            trans.data = this->data;
            return trans;
        }

        //-------------Tensor_part2:
        //-------------MATH:
        //---add
//        template <typename U>
//        ts::Tensor<T>& add(const ts::Tensor<U>& other) const {
//            ts::Tensor<T> result = zeros(other.shape);
//            for (size_t i = 0; i < result.size; ++i) {
//                result.data[i] = this->data[i] + static_cast<T>(other.data[i]);
//            }
//            return *result;
//        }

        static Tensor<T> add(const Tensor<T>&x,const Tensor<T>& y){
            broadcast_tensor<T> processed = broadcast(x, y);
            Tensor<T> input = *(processed.tensor1);
            Tensor<T> other = *(processed.tensor2);
            Tensor<T> result = zeros(other.shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) = *(input.data[i]) + static_cast<T>(*other.data[i]);
            }
            return result;
        }

        Tensor<T> add(const Tensor<T>& y)const{
            broadcast_tensor<T> processed = broadcast(*this, y);
            Tensor<T> input = *(processed.tensor1);
            Tensor<T> other = *(processed.tensor2);
            Tensor<T> result = zeros(other.shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) = *(input.data[i]) + static_cast<T>(*(other.data[i]));
            }
            return result;
        }


        Tensor<T> operator+(const Tensor<T>& other) const {
            return add(other);
        }


        static Tensor<T> add(const Tensor<T>&input,const T& value){
            Tensor<T> result = zeros(input->shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i])= *(input.data[i])+ value;
            }
            return result;
        }

        Tensor<T> add(const T& value) const {
            Tensor<T> result = zeros(this->shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i])= *(this->data[i]) + value;
            }
            return result;
        }

        Tensor<T> operator+(const T& value) const {
            return add(value);
        }



        //----sub

        static Tensor<T> sub(const Tensor<T>& x,const Tensor<T>& y){
            broadcast_tensor<T> processed = broadcast(x, y);
            Tensor<T> input = *(processed.tensor1);
            Tensor<T> other = *(processed.tensor2);
            Tensor<T> result = zeros(other.shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) = *(input.data[i]) - static_cast<T>(*(other.data[i]));
            }
            return result;
        }

        Tensor<T> sub(const Tensor<T>& y)const{
            broadcast_tensor<T> processed = broadcast(*this, y);
            Tensor<T> input = *(processed.tensor1);
            Tensor<T> other = *(processed.tensor2);
            Tensor<T> result = zeros(other.shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) = *(input.data[i]) - static_cast<T>(*(other.data[i]));
            }
            return result;
        }

        Tensor<T> operator-(const Tensor<T>& other) const {
            return sub(other);
        }

        static Tensor<T> sub(const Tensor<T>& input,const T& value) {
            Tensor<T> result = zeros(input->shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) = *(input->data[i]) - value;
            }
            return result;
        }

        Tensor<T> sub(const T& value) const {
            Tensor<T> result = zeros(this->shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) = *(this->data[i]) - value;
            }
            return result;
        }


        Tensor<T> operator-(const T& value) const {
            return sub(value);
        }

        static Tensor<T> openMp_matrix_mul(const Tensor<T>&input,const Tensor<T>&other){
            if(input.dims!=2||other.dims!=2||input.shape[0]!=other.shape[1]||
               input.shape[1]!=other.shape[0]){
                throw ShapeError("ShapeError: Shape does not match");
            }
            size_t rows = input.shape[0];
            size_t cols = other.shape[1];
            size_t commonDim = input.shape[1];
            Tensor<T> result = zeros({rows, cols});
#pragma omp parallel for
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    T sum = 0;
                    for (size_t k = 0; k < commonDim; ++k) {
                        sum += *(input.data[i * commonDim + k]) * *(other.data[k * cols + j]);
                    }
                    *(result.data[i * cols + j]) = sum;
                }
            }
            return result;
        }
        Tensor<T> openMp_matrix_mul(const Tensor<T>& other){
            return openMp_matrix_mul(*this,other);
        }

        static Tensor<T> matrix_mul(const Tensor<T>&input,const Tensor<T>& other){
            if(input.dims!=2||other.dims!=2||input.shape[1]!=other.shape[0]){
                throw ShapeError("ShapeError: Shape does not match");
            }
            size_t rows = input.shape[0];
            size_t cols = other.shape[1];
            size_t commonDim = input.shape[1];
            Tensor<T> result = zeros({rows, cols});
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    T sum = 0;
                    for (size_t k = 0; k < commonDim; ++k) {
                        sum += *(input.data[i * commonDim + k]) * *(other.data[k * cols + j]);
                    }
                    *(result.data[i * cols + j]) = sum;
                }
            }
            return result;
        }

        Tensor<T> matrix_mul(const Tensor<T>& other){
            return matrix_mul(*this,other);
        }

        static Tensor<T> mat_vec_mul(const Tensor<T>&input,const Tensor<T>& other){
            if(input.dims!=2||other.dims!=1||input.shape[1]!=other.shape[0]){
                throw ShapeError("ShapeError: Shape does not match");
            }
            size_t rows = input.shape[0];
            size_t commonDim = input.shape[1];
            Tensor<T> result = zeros({rows});
#pragma omp parallel for
            for (size_t i = 0; i < rows; ++i) {
                T sum = 0;
                for (size_t k = 0; k < commonDim; ++k) {
                    sum += *(input.data[i * commonDim + k]) * *(other.data[k]);
                }
                *(result.data[i]) = sum;
            }
            return result;
        }

        static Tensor<T> openMp_mul(const Tensor<T>&input,const Tensor<T>& other){
            Tensor<T> result = zeros(other.shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) =*(input.data[i]) * static_cast<T>(*(other.data[i]));
            }
            return result;
        }
        Tensor<T> openMp_mul(const Tensor<T>& other){
            return openMp_mul(*this,other);
        }


        static Tensor<T> noBroadcast_mul(const Tensor<T>&x,const Tensor<T>& y){
            Tensor<T> result = zeros(x.shape);
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) =*(x.data[i]) * static_cast<T>(*(y.data[i]));
            }
            return result;
        }

        //mul
        static Tensor<T> mul(const Tensor<T>&x,const Tensor<T>& y){
            broadcast_tensor<T> processed = broadcast(x, y);
            Tensor<T> input = *(processed.tensor1);
            Tensor<T> other = *(processed.tensor2);
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) =*(input.data[i]) * static_cast<T>(*(other.data[i]));
            }
            return result;
        }

        Tensor<T> mul(const Tensor<T>& y)const{
            broadcast_tensor<T> processed = broadcast(*this, y);
            Tensor<T> input = *(processed.tensor1);
            Tensor<T> other = *(processed.tensor2);
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) =*(input.data[i]) * static_cast<T>(*(other.data[i]));
            }
            return result;
        }

        Tensor<T> operator*(const Tensor<T>& other) const {
            return mul(other);
        }

        static Tensor<T> mul(const Tensor<T>&input,const T& value){
            Tensor<T> result = zeros(input->shape);
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i])= *(input.data[i]) * value;
            }
            return result;
        }

        Tensor<T> mul(const T& value) const {
            Tensor<T> result = zeros(this->shape);
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i])= *(this->data[i]) * value;
            }
            return result;
        }

        Tensor<T> operator*(const T& value) const {
            return mul(value);
        }

        //div
        static Tensor<T> div(const Tensor<T>&x,const Tensor<T>& y){
            broadcast_tensor<T> processed = broadcast(x, y);
            Tensor<T> input = *(processed.tensor1);
            Tensor<T> other = *(processed.tensor2);
            Tensor<T> result = zeros(other.shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) =*(input.data[i]) / static_cast<T>(*(other.data[i]));
            }
            return result;
        }

        Tensor<T> div(const Tensor<T>& y)const{
            broadcast_tensor<T> processed = broadcast(*this, y);
            Tensor<T> input = *(processed.tensor1);
            Tensor<T> other = *(processed.tensor2);
            Tensor<T> result = zeros(other.shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i]) =*(input.data[i]) / static_cast<T>(*(other.data[i]));
            }
            return result;
        }


        Tensor<T> operator/(const Tensor<T>& other) const {
            return div(other);
        }


        static Tensor<T> div(const Tensor<T>&input,const T& value){
            Tensor<T> result = zeros(input->shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i])= *(input.data[i]) / value;
            }
            return result;
        }

        Tensor<T> div(const T& value) const {
            Tensor<T> result = zeros(this->shape);
#pragma omp parallel for
            for (size_t i = 0; i < result.size; ++i) {
                *(result.data[i])= *(this->data[i]) / value;
            }
            return result;
        }

        Tensor<T> operator/(const T& value) const {
            return div(value);
        }

        //log
        static Tensor<T> log(const Tensor<T>& tensor){
            Tensor<T> result = zeros(tensor.shape);
#pragma omp parallel for
            for(size_t i=0;i<result.size;++i){
                *(result.data[i]) = std::log(static_cast<T>(*(tensor.data[i])));
            }
            return result;
        }

//        static Tensor<T> operator_log(const Tensor<T>& tensor) {
//            return log(tensor);
//        }

        //reduction operations
        //sum:

        static Tensor<T> sum(const Tensor<T>& tensor, int dim) {
            // 获取 tensor 的形状和维度
            const std::vector<size_t>& shape = tensor.shape;
            size_t dims = tensor.dims;

            // 创建一个新的张量，形状与 tensor 一致，但在指定的维度上大小为 1
            std::vector<size_t> sumShape;
            for(int i = 0; i < dims; i++){
                if(i != dim) sumShape.push_back(shape[i]);
            }
            Tensor<T> result = zeros(sumShape);
            // 计算每个维度上的 stride
            std::vector<size_t> strides(dims);
            size_t stride = 1;
            for (int i = dims - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            // 遍历 tensor 的元素，对指定维度上的元素进行求和
            for (size_t i = 0; i < tensor.size; ++i) {
                // 计算当前元素在 result 中的索引
                size_t preDim = strides[dim] * shape[dim];
                size_t resultIndex = (i % strides[dim]) + (i / preDim) * strides[dim];
                *(result.data[resultIndex]) += *(tensor.data[i]);
            }
            return result;
        }

        Tensor<T> sum(int dim) {
            return sum(*this,dim);
        }

        static Tensor<T> sum(const Tensor<T>& tensor) {
            size_t total = tensor.size;
            T sum = *(tensor.data[0]);
            for (size_t i = 1; i < total; ++i) {
                sum += *(tensor.data[i]);
            }
            T temp[1] = {sum};
            return Tensor<T>(temp);
        }

        Tensor<T> sum(){
            return sum(*this);
        }

        static Tensor<T> mean(const Tensor<T>& tensor, int dim) {
            // 获取 tensor 的形状和维度
            const std::vector<size_t>& shape = tensor.shape;
            size_t dims = tensor.dims;

            // 创建一个新的张量，形状与 tensor 一致，但在指定的维度上大小为 1
            std::vector<size_t> sumShape;
            for(int i = 0; i < dims; i++){
                if(i != dim) sumShape.push_back(shape[i]);
            }
            Tensor<T> result = zeros(sumShape);
            // 计算每个维度上的 stride
            std::vector<size_t> strides(dims);
            size_t stride = 1;
            for (int i = dims - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            // 遍历 tensor 的元素，对指定维度上的元素进行求和
            std::unordered_map<size_t ,int> unorderedMap;
            for (size_t i = 0; i < tensor.size; ++i) {
                // 计算当前元素在 result 中的索引
                size_t preDim = strides[dim] * shape[dim];
                size_t resultIndex = (i % strides[dim]) + (i / preDim) * strides[dim];
                *(result.data[resultIndex]) += *(tensor.data[i]);
                unorderedMap[resultIndex]+=1;
            }
            for(auto P :unorderedMap){
                *(result.data[P.first]) /= P.second;
            }
            return result;
        }

        Tensor<T> mean(int dim) {
            // 获取 tensor 的形状和维度
            return mean(*this,dim);
        }


        static Tensor<T> max(const Tensor<T>& tensor, int dim) {
            return max_or_min(tensor,dim, true);
        }

        Tensor<T> max(int dim) {
            // 获取 tensor 的形状和维度
            return max(*this,dim);
        }

        static Tensor<T> min(const Tensor<T>&tensor,int dim){
            return max_or_min(tensor,dim, false);
        }
        Tensor<T> min(int dim) {
            // 获取 tensor 的形状和维度
            return min(*this,dim);
        }

        //reduction operations
        //sum:



        //sum:

        //comparison operations
        // equal:
        //comparison operations
        // equal:
        static Tensor<bool> eq(const Tensor<T>&x,const Tensor<T>&y){
            Tensor<bool> res= Tensor<bool>::zeros(x.shape);
            for(int i=0;i<x.size;i++){
                *(res.data[i])=(*(x.data[i])==*(y.data[i]));
            }
            return res;
        }

        Tensor<bool> eq(const Tensor<T>&other)const{
            return eq(*this,other);
        }

        Tensor<bool> operator==(const Tensor<T>&other)const{
            return eq(*this,other);
        }

        //ne
        static Tensor<bool> ne(const Tensor<T>&x,const Tensor<T>&y){
            Tensor<bool> res= Tensor<bool>::zeros(x.shape);
            for(int i=0;i<x.size;i++){
                *(res.data[i])=(*(x.data[i])!=*(y.data[i]));
            }
            return res;
        }

        Tensor<bool> ne(const Tensor<T>&other)const{
            ts::Tensor<bool> res= Tensor<bool>::zeros(this->shape);
            for(int i=0;i<this->size;i++){
                *(res.data[i])= *(this->data[i]) != *(other.data[i]);
            }
            return res;
        }

        Tensor<bool> operator!=(const Tensor<T>&other)const{
            return ne(other);
        }

        //gt
        static Tensor<bool> gt(const Tensor<T>&x,const Tensor<T>&y){
            Tensor<bool> res= Tensor<bool>::zeros(x.shape);
            for(int i=0;i<x.size;i++){
                *(res.data[i])=(*(x.data[i]) > *(y.data[i]));
            }
            return res;
        }

        Tensor<bool> gt(const Tensor<T>&other)const{
            return gt(*this,other);
        }

        Tensor<bool> operator>(const Tensor<T>&other)const{
            return gt(*this,other);
        }

        //ge
        static Tensor<bool> ge(const Tensor<T>&x,const Tensor<T>&y){
            Tensor<bool> res= Tensor<bool>::zeros(x.shape);
            for(int i=0;i<x.size;i++){
                *(res.data[i])=(*(x.data[i]) >= *(y.data[i]));
            }
            return res;
        }

        Tensor<bool> ge(const Tensor<T>&other)const{
            return ge(*this,other);
        }

        Tensor<bool> operator>=(const Tensor<T>&other)const{
            return ge(*this,other);
        }

        //lt
        static Tensor<bool> lt(const Tensor<T>&x,const Tensor<T>&y){
            Tensor<bool> res= Tensor<bool>::zeros(x.shape);
            for(int i=0;i<x.size;i++){
                *(res.data[i])=(*(x.data[i]) < *(y.data[i]));
            }
            return res;
        }

        Tensor<bool> lt(const Tensor<T>&other)const{
            return lt(*this,other);
        }

        Tensor<bool> operator<(const Tensor<T>&other)const{
            return lt(*this,other);
        }

        //le
        static Tensor<bool> le(const Tensor<T>&x,const Tensor<T>&y){
            Tensor<bool> res= Tensor<bool>::zeros(x.shape);
            for(int i=0;i<x.size;i++){
                *(res.data[i])=(*(x.data[i]) <= *(y.data[i]));
            }
            return res;
        }

        Tensor<bool> le(const Tensor<T>&other)const{
            return le(*this,other);
        }

        Tensor<bool> operator<=(const Tensor<T>&other)const{
            return le(*this,other);
        }

        // 3.4 einsum
        static Tensor<T> einsum(const std::string expression, std::vector<Tensor<T>> tensors){
            std::regex diagonal_pattern("[\\s]*([a-zA-Z])\\1[\\s]*->[\\s]*\\1[\\s]*");
            std::regex transpose_pattern("[\\s]*([a-zA-Z])(?!\\1)([a-zA-Z])[\\s]*->[\\s]*\\2\\1[\\s]*");
            std::regex permutation_pattern("[\\s]*\\.{3}([a-zA-Z])(?!\\1)([a-zA-Z])[\\s]*->[\\s]*\\.{3}\\2\\1[\\s]*");
            std::regex reduced_sum_pattern("[\\s]*([a-zA-Z])(?!\\1)[a-zA-Z][\\s]*->[\\s]*");
            std::regex reduced_sum_1_pattern("[\\s]*([a-zA-Z])(?!\\1)([a-zA-Z])[\\s]*->[\\s]*\\1[\\s]*");
            std::regex reduced_sum_0_pattern("[\\s]*([a-zA-Z])(?!\\1)([a-zA-Z])[\\s]*->[\\s]*\\2[\\s]*");
            std::regex elementwise_pattern("[\\s]*([a-zA-Z])[\\s]*,[\\s]*\\1[\\s]*->[\\s]*\\1[\\s]*");
            std::regex mat_vec_pattern("[\\s]*([a-zA-Z])(?!\\1)([a-zA-Z])[\\s]*,[\\s]*\\2[\\s]*->[\\s]*\\1[\\s]*");
            std::regex mat_mat_pattern("[\\s]*([a-zA-Z])(?!\\1)([a-zA-Z])[\\s]*,[\\s]*\\2(?!\\1|\\2)([a-zA-Z])[\\s]*->[\\s]*\\1\\3[\\s]*");
            std::regex dot_pattern("[\\s]*([a-zA-Z])[\\s]*,[\\s]*\\1[\\s]*->[\\s]*");
            std::regex pointwise_mul_reduce_sum("[\\s]*([a-zA-Z])(?!\\1)([a-zA-Z])[\\s]*,[\\s]*\\1\\2[\\s]*->[\\s]*");
            std::regex outer_pattern("[\\s]*([a-zA-Z])[\\s]*,[\\s]*(?!\\1)([a-zA-Z])[\\s]*->[\\s]*\\1\\2[\\s]*");
            std::regex bmm_pattern("[\\s]*([a-zA-Z])(?!\\1)([a-zA-Z])(?!\\2)([a-zA-Z])[\\s]*,[\\s]*\\1\\3(?!\\3)([a-zA-Z])[\\s]*->[\\s]*\\1\\2\\4[\\s]*");
            std::regex bilinear_pattern("[\\s]*([a-zA-Z])(?!\\1)([a-zA-Z])[\\s]*,[\\s]*(?!\\1|\\2)([a-zA-Z])\\2(?!\\1|\\2|\\3)([a-zA-Z])[\\s]*,[\\s]*\\1\\4[\\s]*->[\\s]*\\1\\3[\\s]*");

            std::regex single_input_pattern("[^,]*");
            std::regex double_input_pattern("[^,]*,[^,]*");
            std::regex trible_input_pattern("[^,]*,[^,]*,[^,]*");
            std::vector<size_t> test = {1};
            if(tensors.size() < 1 && std::regex_match(expression, single_input_pattern)){
                // throw
                throw EmptyVectorError("EmptyVectorError: Please input at least 1 tensor");
            }
            if(std::regex_match(expression, diagonal_pattern)){
                //                std::cout << "diag match" << std::endl;
                return Tensor<T>::diagonal(tensors[0]);
            }
            if(std::regex_match(expression, transpose_pattern)){
                //                std::cout << "transpose match" << std::endl;
                return tensors[0].transpose(0, 1);
            }
            if(std::regex_match(expression, permutation_pattern)){
                //                std::cout << "permute match" << std::endl;
                std::vector<size_t> shape = tensors[0].shape;
                size_t end = shape.size() - 1;
                std::vector<size_t> dims;
                for(size_t i = 0; i < shape.size() - 2; i++){
                    dims.push_back(i);
                }
                dims.push_back(end);
                dims.push_back(end - 1);
                return Tensor<T>::permute(tensors[0], dims);
            }
            if(std::regex_match(expression, reduced_sum_pattern)){
                //                std::cout << "reduced sum match" << std::endl;
                return Tensor<T>::sum(tensors[0]);
            }
            if(std::regex_match(expression, reduced_sum_0_pattern)){
                //                std::cout << "reduced sum 0 match" << std::endl;
                return Tensor<T>::sum(tensors[0], 0);
            }
            if(std::regex_match(expression, reduced_sum_1_pattern)){
                //                std::cout << "reduced sum 1 match" << std::endl;
                return Tensor<T>::sum(tensors[0], 1);
            }
            if(tensors.size() < 2 && std::regex_match(expression, double_input_pattern)){
                // throw exception
                throw ExpressionMismatchError("ExpressionMismatchError: Numbers of input tensors cannot match the expression");
            }
            if(std::regex_match(expression, mat_vec_pattern)){
                return Tensor<T>::mat_vec_mul(tensors[0], tensors[1]);
            }
            if(std::regex_match(expression, mat_mat_pattern)){
                //                std::cout << "mat mul match" << std::endl;
                return Tensor<T>::matrix_mul(tensors[0], tensors[1]);
            }
            if(std::regex_match(expression, dot_pattern)){
//                std::cout << "dot match" << std::endl;
                return Tensor<T>::dot(tensors[0], tensors[1]);
            }
            if(std::regex_match(expression, pointwise_mul_reduce_sum)){
                //                std::cout << "pointwise mul match" << std::endl;
                Tensor<T> temp = Tensor<T>::mul(tensors[0], tensors[1]);
                return Tensor<T>::sum(temp);
            }
            if(std::regex_match(expression, elementwise_pattern)) {
                //                std::cout << "element match" << std::endl;
                return tensors[0] * tensors[1];
            }
            if(std::regex_match(expression, outer_pattern)){
//                std::cout << "outer match" << std::endl;
                return outer(tensors[0], tensors[1]);
            }
            if(std::regex_match(expression, bmm_pattern)){
//                std::cout << "bmm match" << std::endl;
                return bmm(tensors[0], tensors[1]);
            }
            if(tensors.size() < 3 && std::regex_match(expression, trible_input_pattern)){
                throw ExpressionMismatchError("ExpressionMismatchError: Numbers of input tensors cannot match the expression");
            }
            if(std::regex_match(expression, bilinear_pattern)){
                return bilinear(tensors[0], tensors[1], tensors[2]);
            }
            // throw exception no matches
            throw ExpressionNotSupportedError("ExpressionNotSupportedError: Expression not supported");
        }

        //outer product:
        static Tensor<T> outer(const Tensor<T>&x,const Tensor<T>&y){
            if(x.dims!=1||y.dims!=1){
                throw ShapeError("ShapeError: Shape are not allowed.");
            }
            size_t n=x.shape[0];
            size_t m=y.shape[0];
            std::vector<size_t>res_shape;
            res_shape.push_back(n);
            res_shape.push_back(m);
            Tensor<T>result= zeros(res_shape);
            for(int i=0;i<n;i++){
                for(int j=0;j<m;j++){
                    *(result.data[i*m+j]) = *(x.data[i]) * *(y.data[j]);
                }
            }
            return result;
        }

        Tensor<T> outer(const Tensor<T>&other){
            return outer(*this,other);
        }

        //batch matrix multiplication
        static Tensor<T> bmm(const Tensor<T>&x,const Tensor<T>&y){
            if(x.dims!=3||y.dims!=3||x.shape[0]!=y.shape[0]||x.shape[2]!=y.shape[1]){
                throw ShapeError("ShapeError: Shape are not allowed.");
            }
            size_t b=x.shape[0];
            size_t n=x.shape[1];
            size_t m=x.shape[2];
            size_t p=y.shape[2];
            std::vector<size_t>res_shape;
            res_shape.push_back(b);
            res_shape.push_back(n);
            res_shape.push_back(p);
            Tensor<T>result= zeros(res_shape);
            for (size_t i = 0; i < b; i++) {
                for (size_t j = 0; j < n; j++) {
                    for (size_t k = 0; k < p; k++) {
                        for (size_t l = 0; l < m; l++) {
                            *(result.data[(i * n + j) * p + k]) += *(x.data[(i * n + j) * m + l]) * *(y.data[(i * m + l) * p + k]);
                        }
                    }
                }
            }
            return result;
        }

        Tensor<T> bmm(const Tensor<T>&other){
            return bmm(*this,other);
        }

        static Tensor<T> dot(const Tensor<T>&x,const Tensor<T>&y){
            if(x.dims!=1||y.dims!=1||x.shape[0]!=y.shape[0]){
                throw ShapeError("ShapeError: Shape are not allowed.");
            }
            std::vector<size_t>v_shape;
            v_shape.push_back(1);
            Tensor<T> a=zeros(v_shape);
            for(int i=0;i<x.size;i++){
                *(a.data[0]) += *(x.data[i]) * *(y.data[i]);
            }
            return a;
        }

        Tensor<T> dot(const Tensor<T>&other){
            return dot(*this,other);
        }

        static Tensor<T> diagonal(const Tensor<T>& input, int offset = 0) {
            size_t inputDims = input.dims;
            if (inputDims < 2) {
                throw DimensionError("DimensionError: Input tensor must have at least 2 dimensions.");
            }

            size_t rows = input.shape[inputDims - 2];
            size_t cols = input.shape[inputDims - 1];

            if (offset > 0) {
                size_t diagSize = std::min(rows, cols - offset);
                std::vector<size_t> diagShape = { diagSize };
                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);

                for (size_t i = 0; i < diagSize; ++i) {
                    *(diagTensor.data[i]) = *(input.data[i * input.strides[inputDims - 2] + (i + offset) * input.strides[inputDims - 1]]);
                }

                return diagTensor;
            } else if (offset < 0) {
                size_t diagSize = std::min(rows + offset, cols);
                std::vector<size_t> diagShape = { diagSize };
                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);

                for (size_t i = 0; i < diagSize; ++i) {
                    *(diagTensor.data[i]) = *(input.data[(i - offset) * input.strides[inputDims - 2] + i * input.strides[inputDims - 1]]);
                }

                return diagTensor;
            } else {
                size_t diagSize = std::min(rows, cols);
                std::vector<size_t> diagShape = { diagSize };
                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);

                for (size_t i = 0; i < diagSize; ++i) {
                    *(diagTensor.data[i]) = *(input.data[i * input.strides[inputDims - 2] + i * input.strides[inputDims - 1]]);
                }

                return diagTensor;
            }
        }

        static Tensor<T> bilinear(Tensor<T> t1, Tensor<T> t2, Tensor<T> t3){
            if(t1.dims != 2 || t2.dims != 3 || t3.dims != 2){
                throw ShapeError("ShapeError: Shape mismatch");
            }
            if(t1.shape[0] != t3.shape[0] || t1.shape[1] != t2.shape[1] || t2.shape[2] != t3.shape[1]){
                throw ShapeError("ShapeError: Shape mismatch");
            }
            std::vector<size_t> newDimension;
            newDimension.push_back(t1.shape[0]);
            newDimension.push_back(t2.shape[0]);
            Tensor<T> result = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < result.shape[0]; i++){
                for(size_t j = 0; j < result.shape[1]; j++){
                    size_t src_index = i * result.strides[0] + j * result.strides[1];
                    T sum = 0;
                    for(size_t k = 0; k < t1.shape[1]; k++){
                        for(size_t l = 0; l < t2.shape[2]; l++){
                            size_t x_index = i * t1.strides[0] + k * t1.strides[1];
                            size_t A_index = j * t2.strides[0] + k * t2.strides[1] + l * t2.strides[2];
                            size_t y_index = i * t3.strides[0] + l * t3.strides[1];
                            sum += *(t1.data[x_index]) * (*(t2.data[A_index])) * (*(t3.data[y_index]));
                        }
                    }
                    *(result.data[src_index]) = sum;
                }
            }
            return result;
        }


        static void save(const Tensor<T>& tensor, const std::string& filename) {
            std::ofstream file(filename, std::ios::out | std::ios::binary);
            if (file.is_open()) {
                // 设置文件流的locale为UTF-8
                file.imbue(std::locale(""));

                // 保存Tensor的维度信息
                size_t dims = tensor.dims;
                file.write(reinterpret_cast<const char*>(&dims), sizeof(size_t));

                // 保存Tensor的shape信息
                file.write(reinterpret_cast<const char*>(tensor.shape.data()), tensor.shape.size() * sizeof(size_t));

                // 保存Tensor的数据
                for (const auto& sharedPtr : tensor.data) {
                    const T* rawData = sharedPtr.get();
                    file.write(reinterpret_cast<const char*>(rawData), sizeof(T));
                }

                file.close();
            } else {
                throw OpenFileError("OpenFileError: Failed to open file: " + filename);
            }
        }

        static Tensor<T> load(const std::string& filename) {
            std::ifstream file(filename, std::ios::in | std::ios::binary);
            if (file.is_open()) {
                // 设置文件流的locale为UTF-8
                file.imbue(std::locale(""));

                // 读取Tensor的维度信息
                size_t dims;
                file.read(reinterpret_cast<char*>(&dims), sizeof(size_t));

                // 读取Tensor的shape信息
                std::vector<size_t> shape(dims);
                file.read(reinterpret_cast<char*>(shape.data()), dims * sizeof(size_t));

                // 读取Tensor的数据
                size_t size = 1;
                for (size_t dim : shape) {
                    size *= dim;
                }
                std::vector<std::shared_ptr<T>> data(size);
                for (auto& sharedPtr : data) {
                    sharedPtr = std::make_shared<T>();
                    file.read(reinterpret_cast<char*>(sharedPtr.get()), sizeof(T));
                }
                file.close();
                return Tensor<T>(data, size, shape);
            } else {
                throw OpenFileError("OpenFileError: Failed to open file: " + filename);
            }
        }





    private:
        static size_t printTensor(std::ostream& os, const Tensor<T>& tensor, size_t level, size_t offset = 0,size_t block_nums=1) {
            size_t dimSize = tensor.shape[level];
            if(typeid(bool) != typeid(T)){
                for (size_t i = 0; i < dimSize; ++i) {
                    if (level < tensor.dims - 1) {
                        os << "[";
                        block_nums++;
                        block_nums = printTensor(os, tensor, level + 1, offset + i * tensor.strides[level], block_nums);
                        block_nums--;
                        os << "]";
                        if (i < dimSize - 1) {
                            os << ", ";
                            os << std::endl;
                            for (int l = 0; l < block_nums; l++) {
                                os << " ";
                            }
                        }
                    } else {
                        T tmp = *(tensor.data[i + offset].get());
                        os << *(tensor.data[i + offset]);
                        if (i < dimSize - 1) {
                            os << ", ";
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < dimSize; ++i) {
                    if (level < tensor.dims - 1) {
                        os << std::boolalpha << "[";
                        block_nums++;
                        block_nums = printTensor(os, tensor, level + 1, offset + i * tensor.strides[level], block_nums);
                        block_nums--;
                        os << std::boolalpha << "]";
                        if (i < dimSize - 1) {
                            os << std::boolalpha << ", ";
                            os << std::boolalpha << std::endl;
                            for (int l = 0; l < block_nums; l++) {
                                os << std::boolalpha << " ";
                            }
                        }
                    } else {
                        os << std::boolalpha << *(tensor.data[i + offset]);
                        if (i < dimSize - 1) {
                            os << std::boolalpha << ", ";
                        }
                    }
                }
            }
            return block_nums;
        }
        static Tensor<T> max_or_min(const Tensor<T>& tensor, int dim,bool larger) {
            // 获取 tensor 的形状和维度
            const std::vector<size_t>& shape = tensor.shape;
            size_t dims = tensor.dims;

            // 创建一个新的张量，形状与 tensor 一致，但在指定的维度上大小为 1
            std::vector<size_t> sumShape;
            for(int i = 0; i < dims; i++){
                if(i != dim) sumShape.push_back(shape[i]);
            }
            Tensor<T> result = zeros(sumShape);
            // 计算每个维度上的 stride
            std::vector<size_t> strides(dims);
            size_t stride = 1;
            for (int i = dims - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            // 遍历 tensor 的元素，对指定维度上的元素进行求和
            std::unordered_map<size_t ,int> unorderedMap;
            for (size_t i = 0; i < tensor.size; ++i) {
                // 计算当前元素在 result 中的索引
                size_t preDim = strides[dim] * shape[dim];
                size_t resultIndex = (i % strides[dim]) + (i / preDim) * strides[dim];
                if(unorderedMap.count(resultIndex)){
                    if(larger) {
                        unorderedMap[resultIndex] = *(tensor.data[i]) > unorderedMap[resultIndex] ? *(tensor.data[i]) : unorderedMap[resultIndex];
                    }else{
                        unorderedMap[resultIndex] = *(tensor.data[i]) < unorderedMap[resultIndex] ? *(tensor.data[i]) : unorderedMap[resultIndex];
                    }
                }else{
                    unorderedMap[resultIndex]=*(tensor.data[i]);
                }
            }
            for(auto P :unorderedMap){
                *(result.data[P.first])=P.second;
            }
            return result;
        }

        friend broadcast_tensor<T>;
    };
}// namespace ts