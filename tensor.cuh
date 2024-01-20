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
#include <stdio.h>

namespace ts {
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
                std::shared_ptr<T> tmp(new T[this->size](), std::default_delete<T[]>());
                this->data = tmp;
            }
            this->data.get()[offset/sizeof(T)] = data;
            return 1;
        }

        void randInt(const std::vector<size_t>& dimensions) {
            size_t totalSize = 1;
            for (size_t dim : dimensions) {
                totalSize *= dim;
            }
            for (size_t i = 0; i < totalSize; ++i) {
                this->data.get()[i] = std::rand() % 100;
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
                this->data.get()[i] = r;
            }
        }

    public:
        std::shared_ptr<T> data;
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

        Tensor(std::shared_ptr<T> data, size_t size, const std::vector<size_t>& dimensions) {
            this->data = data;
            this->size = size;
            this->shape = dimensions;

            this->dims = dimensions.size();
            this->strides.resize(this->dims);

            for (size_t i = 0; i < this->dims; ++i) {
                size /= dimensions[i];
                this->strides[i] = size;
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
            std::shared_ptr<T> tmp(new T[totalSize](), std::default_delete<T[]>());
            randomTensor.data = tmp;
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
            std::shared_ptr<T> tmp(new T[totalSize](), std::default_delete<T[]>());
            zeroTensor.data = tmp;
            for(int i = 0; i < totalSize; i++) zeroTensor.data.get()[i] = 0;

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

            std::shared_ptr<T> tmp(new T[totalSize](), std::default_delete<T[]>());
            oneTensor.data = tmp;
            for(int i = 0; i < totalSize; i++) oneTensor.data.get()[i] = 1;

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
            std::shared_ptr<T> tmp(new T[totalSize](), std::default_delete<T[]>());
            fullTensor.data = tmp;
            for(int i = 0; i < totalSize; i++) fullTensor.data.get()[i] = value;

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
            std::shared_ptr<T> tmp(new T[totalSize](), std::default_delete<T[]>());
            eyeTensor.data = tmp;
            for (size_t i = 0; i < totalSize; ++i) {
                if(i / row == i % row) eyeTensor.data.get()[i] = 1;
                else eyeTensor.data.get()[i] = 0;
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

//        static Tensor<T> diagonal(const Tensor<T>& input, int offset = 0) {
//            size_t inputDims = input.dims;
//            if (inputDims < 2) {
//                throw std::runtime_error("输入的tensor至少包含两个维度");
//            }
//
//            size_t rows = input.shape[inputDims - 2];
//            size_t cols = input.shape[inputDims - 1];
//
//            if (offset > 0) {
//                size_t diagSize = std::min(rows, cols - offset);
//                std::vector<size_t> diagShape = { diagSize };
//                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);
//
//                for (size_t i = 0; i < diagSize; ++i) {
//                    diagTensor.data.get()[i] = input.data.get()[i * input.strides[inputDims - 2] + (i + offset) * input.strides[inputDims - 1]];
//                }
//
//                return diagTensor;
//            } else if (offset < 0) {
//                size_t diagSize = std::min(rows + offset, cols);
//                std::vector<size_t> diagShape = { diagSize };
//                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);
//
//                for (size_t i = 0; i < diagSize; ++i) {
//                    diagTensor.data.get()[i] = input.data.get()[(i - offset) * input.strides[inputDims - 2] + i * input.strides[inputDims - 1]];
//                }
//
//                return diagTensor;
//            } else {
//                size_t diagSize = std::min(rows, cols);
//                std::vector<size_t> diagShape = { diagSize };
//                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);
//
//                for (size_t i = 0; i < diagSize; ++i) {
//                    diagTensor.data.get()[i] = input.data.get()[i * input.strides[inputDims - 2] + i * input.strides[inputDims - 1]];
//                }
//
//                return diagTensor;
//            }
//        }


        Tensor<T> operator()(size_t index) {
            int indexDim = this->dims - 1;
            std::vector<size_t> dimensions;
            for(int i = 1; i <= indexDim; i++) dimensions.push_back(this->shape[i]);
            if(dimensions.empty()) dimensions.push_back(1);
            int indexSize = this->size / shape[0];
            std::shared_ptr<T> indexData(this->data, this->data.get() + index * this->strides[0]);
            Tensor<T> indexT(indexData, indexSize, dimensions);
            return indexT;
        }

        Tensor<T> operator()(size_t location, const std::vector<size_t>& indices) {
            int sliceSize = indices[1] - indices[0];
            int startIndex = indices[0];
            std::vector<size_t> dimensions;
            dimensions.push_back(sliceSize);
            std::shared_ptr<T> sliceData(this->data, this->data.get() + location * this->strides[0] + startIndex);
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


        static std::vector<size_t> get_broadcastShape(const Tensor<T>&x,const Tensor<T>&y){
            //对比x,y的维度是否符合广播条件
            int x_end=x.shape.size()-1;
            int y_end=y.shape.size()-1;
            while(x_end>=0&&y_end>=0){
                if(!(x.shape[x_end]==y.shape[y_end]||x.shape[x_end]==1||y.shape[y_end]==1)){
                    throw std::invalid_argument("Shape are not broadcastable.");
                }
                x_end--;
                y_end--;
            }
            //检查完毕，开始广播：
            std::vector<size_t> broadcast_Shape;
            int max_dim = std::max(x.shape.size(),y.shape.size());
            broadcast_Shape.resize(max_dim);
            int x_idx = x.shape.size()-1;
            int y_idx = y.shape.size()-1;
            int broadcast_idx = max_dim -1;
            while(broadcast_idx >=0){
                size_t dim_x = (x_idx>=0) ? x.shape[x_idx] : 1;
                size_t dim_y = (y_idx>=0) ? y.shape[y_idx] : 1;
                broadcast_Shape[broadcast_idx] = std::max(dim_x,dim_y);
                x_idx--;
                y_idx--;
                broadcast_idx--;
            }
            return broadcast_Shape;
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
                T value = x.data.get()[0];  // 默认使用 x 张量中的第一个元素
                for (int j = 0; j < xDims; ++j) {
                    if (xShape[j] > 1) {
                        value = x.data.get()[xIndices[j]];  // 使用对应位置的 x 元素
                        break;
                    }
                }
                result.data.get()[i] = value;

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
            Tensor<T> mainTensor = tensors[0];
            size_t mainDim = mainTensor.shape[dim];
            std::vector<size_t> newDimension = mainTensor.shape;
            for(size_t i = 1; i < tensors.size(); i++){
                newDimension[dim] += tensors[i].shape[dim];
            }
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++){
                std::shared_ptr<T> dst_ptr(trans.data, trans.data.get() + i);
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
                std::shared_ptr<T> src_ptr(tensors[k].data, tensors[k].data.get() + src_index);
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
                    std::cout << "error" << std::endl;
                } else{
                    newDimension[i] *= dims[i];
                }
            }
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for (size_t i = 0; i < trans.size; ++i) {
                std::shared_ptr<T> dst_ptr(trans.data, trans.data.get() + i);
                size_t src_index = 0;
                size_t dst_index = i;
                for (size_t j = 0; j < trans.strides.size(); ++j) {
                    size_t coordinate = dst_index / trans.strides[j];
                    dst_index %= trans.strides[j];
                    coordinate %= tensor_copy.shape[j];
                    src_index += coordinate * tensor_copy.strides[j];
                }
                std::shared_ptr<T> src_ptr(tensor_copy.data, tensor_copy.data.get() + src_index);
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
                std::shared_ptr<T> data_ptr(this->data, this->data.get() + i);
                *data_ptr = scalar;
            }
        }

        // used for tensor assignment to part of a tensor
        void operator=(Tensor<T> operand){
            size_t length = operand.shape.front();
            for(size_t i = 0; i < length; i++){
                std::shared_ptr<T> dst_ptr(this->data, this->data.get() + i);
                std::shared_ptr<T> src_ptr(operand.data, operand.data.get() + i);
                *dst_ptr = *src_ptr;
            }
        }

        // 2.4 transpose & permutation
        //transpose
        static Tensor<T> transpose(Tensor<T> tensor, int dim1, int dim2) {
            std::vector<size_t> newDimension = tensor.shape;
            newDimension[dim1] = tensor.shape[dim2];
            newDimension[dim2] = tensor.shape[dim1];
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++) {
                std::shared_ptr<T> src_ptr(tensor.data, tensor.data.get() + i);
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
                std::shared_ptr<T> dst_ptr(trans.data, trans.data.get() + dst_index);
                *dst_ptr = *src_ptr;
            }
            return trans;
        }

        Tensor<T> transpose(int dim1, int dim2) {
            std::vector<size_t> newDimension = this->shape;
            newDimension[dim1] = this->shape[dim2];
            newDimension[dim2] = this->shape[dim1];
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++) {
                std::shared_ptr<T> src_ptr(this->data, this->data.get() + i);
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
                std::shared_ptr<T> dst_ptr(trans.data, trans.data.get() + dst_index);
                *dst_ptr = *src_ptr;
            }
            return trans;
        }

        // permutation
        static Tensor<T> permute(Tensor<T> tensor, const std::vector<int> & dim){
            std::vector<size_t> newDimension = tensor.shape;
            for(size_t i = 0; i < dim.size(); i++) newDimension[i] = tensor.shape[dim[i]];
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++) {
                std::shared_ptr<T> src_ptr(tensor.data, tensor.data.get() + i);
                size_t src_index = i;
                size_t dst_index = 0;
                for (size_t j = 0; j < tensor.strides.size(); ++j) {
                    size_t coordinate = src_index / tensor.strides[j];
                    src_index %= tensor.strides[j];
                    dst_index += coordinate * trans.strides[dim[j]];
                }
                std::shared_ptr<T> dst_ptr(trans.data, trans.data.get() + dst_index);
                *dst_ptr = *src_ptr;
            }
            return trans;
        }

        Tensor<T> permute(const std::vector<int> & dim){
            std::vector<size_t> newDimension = this->shape;
            for(size_t i = 0; i < dim.size(); i++) newDimension[i] = this->shape[dim[i]];
            Tensor<T> trans = Tensor<T>::zeros(newDimension);
            for(size_t i = 0; i < trans.size; i++) {
                std::shared_ptr<T> src_ptr(this->data, this->data.get() + i);
                size_t src_index = i;
                size_t dst_index = 0;
                for (size_t j = 0; j < this->strides.size(); ++j) {
                    size_t coordinate = src_index / this->strides[j];
                    src_index %= this->strides[j];
                    dst_index += coordinate * trans.strides[dim[j]];
                }
                std::shared_ptr<T> dst_ptr(trans.data, trans.data.get() + dst_index);
                *dst_ptr = *src_ptr;
            }
            return trans;
        }

        // 2.5 view
        static Tensor<T> view(Tensor<T> tensor, const std::vector<int> shape){
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
                    std::cout << "error" << std::endl;
                }
            }
            if(reserve != -1){
                size_t lost = tensor.size / full_size;
                newDimension[reserve] = lost;
                full_size *= lost;
            }
            if(tensor.size != full_size) {
                //error
                std::cout << "error" << std::endl;
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
                    std::cout << "error" << std::endl;
                }
            }
            if(reserve != -1){
                size_t lost = this->size / full_size;
                newDimension[reserve] = lost;
                full_size *= lost;
            }
            if(this->size != full_size) {
                //error
                std::cout << "error" << std::endl;
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

        static Tensor<T> add(const Tensor<T>&input,const Tensor<T>& other){
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i] =input.data.get()[i] + static_cast<T>(other.data.get()[i]);
            }
            return result;
        }

        Tensor<T> add(const Tensor<T>& other)const{
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i] =this->data.get()[i] + static_cast<T>(other.data.get()[i]);
            }
            return result;
        }


        Tensor<T> operator+(const Tensor<T>& other) const {
            return add(other);
        }


        static Tensor<T> add(const Tensor<T>&input,const T& value){
            Tensor<T> result = zeros(input->shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i]= input.data.get()[i]+ value;
            }
            return result;
        }

        Tensor<T> add(const T& value) const {
            Tensor<T> result = zeros(this->shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i]= this->data.get()[i] + value;
            }
            return result;
        }

        Tensor<T> operator+(const T& value) const {
            return add(value);
        }



        //----sub

        static Tensor<T> sub(const Tensor<T>& input,const Tensor<T>& other){
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i]= input.data.get()[i]- static_cast<T>(other.data.get()[i]);
            }
            return result;
        }

        Tensor<T> sub(const Tensor<T>& other)const{
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i]= this->data.get()[i] - static_cast<T>(other.data.get()[i]);
            }
            return result;
        }

        Tensor<T> operator-(const Tensor<T>& other) const {
            return sub(other);
        }

        static Tensor<T> sub(const Tensor<T>& input,const T& value) {
            Tensor<T> result = zeros(input->shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i] = input->data.get()[i] - value;
            }
            return result;
        }

        Tensor<T> sub(const T& value) const {
            Tensor<T> result = zeros(this->shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i] = this->data.get()[i] - value;
            }
            return result;
        }


        Tensor<T> operator-(const T& value) const {
            return sub(value);
        }

        //mul
        static Tensor<T> mul(const Tensor<T>&input,const Tensor<T>& other){
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i] =input.data.get()[i] * static_cast<T>(other.data.get()[i]);
            }
            return result;
        }

        Tensor<T> mul(const Tensor<T>& other)const{
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i] =this->data.get()[i] * static_cast<T>(other.data.get()[i]);
            }
            return result;
        }


        Tensor<T> operator*(const Tensor<T>& other) const {
            return mul(other);
        }


        static Tensor<T> mul(const Tensor<T>&input,const T& value){
            Tensor<T> result = zeros(input->shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i]= input.data.get()[i]* value;
            }
            return result;
        }

        Tensor<T> mul(const T& value) const {
            Tensor<T> result = zeros(this->shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i]= this->data.get()[i] * value;
            }
            return result;
        }

        Tensor<T> operator*(const T& value) const {
            return mul(value);
        }

        //gpu mul:
//         void matrixMulGPU()


        //div
        static Tensor<T> div(const Tensor<T>&input,const Tensor<T>& other){
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i] =input.data.get()[i] / static_cast<T>(other.data.get()[i]);
            }
            return result;
        }

        Tensor<T> div(const Tensor<T>& other)const{
            Tensor<T> result = zeros(other.shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i] =this->data.get()[i] / static_cast<T>(other.data.get()[i]);
            }
            return result;
        }


        Tensor<T> operator/(const Tensor<T>& other) const {
            return div(other);
        }


        static Tensor<T> div(const Tensor<T>&input,const T& value){
            Tensor<T> result = zeros(input->shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i]= input.data.get()[i]/ value;
            }
            return result;
        }

        Tensor<T> div(const T& value) const {
            Tensor<T> result = zeros(this->shape);
            for (size_t i = 0; i < result.size; ++i) {
                result.data.get()[i]= this->data.get()[i] / value;
            }
            return result;
        }

        Tensor<T> operator/(const T& value) const {
            return div(value);
        }

        //log
        static Tensor<T> log(const Tensor<T>& tensor){
            Tensor<T> result = zeros(tensor.shape);
            for(size_t i=0;i<result.size;++i){
                result.data.get()[i] = std::log(static_cast<T>(tensor.data.get()[i]));
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
                result.data.get()[resultIndex] += tensor.data.get()[i];
            }
            return result;
        }

         Tensor<T> sum( int dim) {
             return sum(*this,dim);
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
                result.data.get()[resultIndex] += tensor.data.get()[i];
                unorderedMap[resultIndex]+=1;
            }
            for(auto P :unorderedMap){
                result.data.get()[P.first]/=P.second;
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
                res.data.get()[i]=(x.data.get()[i]==y.data.get()[i]);
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
                res.data.get()[i]=(x.data.get()[i]!=y.data.get()[i]);
            }
            return res;
        }

        Tensor<bool> ne(const Tensor<T>&other)const{
            ts::Tensor<bool> res= Tensor<bool>::zeros(this->shape);
            for(int i=0;i<this->size;i++){
                res.data.get()[i]= this->data.get()[i] != other.data.get()[i];
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
                res.data.get()[i]=(x.data.get()[i]>y.data.get()[i]);
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
                res.data.get()[i]=(x.data.get()[i]>=y.data.get()[i]);
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
                res.data.get()[i]=(x.data.get()[i]<y.data.get()[i]);
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
                res.data.get()[i]=(x.data.get()[i]<=y.data.get()[i]);
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
        // /^[\s]*([a-zA-Z])[\s]*,[\s]*\1[\s]*->[\s]*$/
        // /^[\s]*([a-zA-Z])[\s]*,[\s]*\1[\s]*->[\s]*\1[\s]*$/
        // /^[\s]*([a-zA-Z]){2}[\s]*->[\s]*\1[\s]*$/
        // /^[\s]*([a-zA-Z])[\s]*,[\s]*([a-zA-Z])[\s]*->[\s]*\1\2[\s]*$/
        // /^[\s]*([a-zA-Z])([a-zA-Z])([a-zA-Z])[\s]*,[\s]*\1\3([a-zA-Z])[\s]*->[\s]*\1\2\4[\s]*$/
        static Tensor<T> einsum(const std::string expression, const Tensor<T> tensor1, Tensor<T> tensor2){
            std::regex dot_pattern("[\\s]*([a-zA-Z])[\\s]*,[\\s]*\\1[\\s]*->[\\s]*");
            std::regex elementwise_pattern("[\\s]*([a-zA-Z])[\\s]*,[\\s]*\\1[\\s]*->[\\s]*\\1[\\s]*");
            std::regex diagonal_pattern("[\\s]*([a-zA-Z]){2}[\\s]*->[\\s]*\\1[\\s]*");
            std::regex outer_pattern("[\\s]*([a-zA-Z])[\\s]*,[\\s]*([a-zA-Z])[\\s]*->[\\s]*\\1\\2[\\s]*");
            std::regex bmm_pattern("[\\s]*([a-zA-Z])([a-zA-Z])([a-zA-Z])[\\s]*,[\\s]*\\1\\3([a-zA-Z])[\\s]*->[\\s]*\\1\\2\\4[\\s]*");
            std::vector<size_t> test = {1};
            if(std::regex_match(expression, dot_pattern)){
                std::cout << "dot match" << std::endl;
                return Tensor<T>::zeros(test);
            }
            if(std::regex_match(expression, elementwise_pattern)){
                std::cout << "element match" << std::endl;
                return tensor1 * tensor2;
            }
            if(std::regex_match(expression, diagonal_pattern)){
                std::cout << "diag match" << std::endl;
                return Tensor<T>::zeros(test);
            }
            if(std::regex_match(expression, outer_pattern)){
                std::cout << "outer match" << std::endl;
                return outer(tensor1, tensor2);
            }
            if(std::regex_match(expression, bmm_pattern)){
                std::cout << "bmm match" << std::endl;
                return bmm(tensor1, tensor2);
            }
            // throw exception no matches
            std::cerr << "expression not supported" << std::endl;
            return Tensor<T>::zeros(test);
        }

        //outer product:
        static Tensor<T> outer(const Tensor<T>&x,const Tensor<T>&y){
            if(x.dims!=2||y.dims!=2||x.shape.size()!=2||y.shape.size()!=2||
                x.shape[0]!=1||y.shape[0]!=1){
                throw std::invalid_argument("Shape are not allowed.");
            }
            size_t n=x.shape[1];
            size_t m=y.shape[1];
            std::vector<size_t>res_shape;
            res_shape.push_back(n);
            res_shape.push_back(m);
            Tensor<T>result= zeros(res_shape);
            for(int i=0;i<n;i++){
                for(int j=0;j<m;j++){
                    result.data.get()[i*m+j]=x.data.get()[i]*y.data.get()[j];
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
                throw std::invalid_argument("Shape are not allowed.");
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
                            result.data.get()[(i * n + j) * p + k] += x.data.get()[(i * n + j) * m + l] * y.data.get()[(i * m + l) * p + k];
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
                throw std::invalid_argument("Shape are not allowed.");
            }
            std::vector<size_t>v_shape;
            v_shape.push_back(1);
            Tensor<T> a=zeros(v_shape);
            for(int i=0;i<x.size;i++){
                a.data.get()[0]+=x.data.get()[i]*y.data.get()[i];
            }
            return a;
        }

        Tensor<T> dot(const Tensor<T>&other){
            return dot(*this,other);
        }

        static Tensor<T> diagonal(const Tensor<T>& input, int offset = 0) {
            size_t inputDims = input.dims;
            if (inputDims < 2) {
                throw std::runtime_error("Input tensor must have at least 2 dimensions.");
            }

            size_t rows = input.shape[inputDims - 2];
            size_t cols = input.shape[inputDims - 1];

            if (offset > 0) {
                size_t diagSize = std::min(rows, cols - offset);
                std::vector<size_t> diagShape = { diagSize };
                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);

                for (size_t i = 0; i < diagSize; ++i) {
                    diagTensor.data.get()[i] = input.data.get()[i * input.strides[inputDims - 2] + (i + offset) * input.strides[inputDims - 1]];
                }

                return diagTensor;
            } else if (offset < 0) {
                size_t diagSize = std::min(rows + offset, cols);
                std::vector<size_t> diagShape = { diagSize };
                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);

                for (size_t i = 0; i < diagSize; ++i) {
                    diagTensor.data.get()[i] = input.data.get()[(i - offset) * input.strides[inputDims - 2] + i * input.strides[inputDims - 1]];
                }

                return diagTensor;
            } else {
                size_t diagSize = std::min(rows, cols);
                std::vector<size_t> diagShape = { diagSize };
                Tensor<T> diagTensor = Tensor<T>::zeros(diagShape);

                for (size_t i = 0; i < diagSize; ++i) {
                    diagTensor.data.get()[i] = input.data.get()[i * input.strides[inputDims - 2] + i * input.strides[inputDims - 1]];
                }

                return diagTensor;
            }
        }



        template <typename U>
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
                file.write(reinterpret_cast<const char*>(tensor.data.get()), tensor.size * sizeof(T));

                file.close();
            } else {
                std::cerr << "Failed to open file: " << filename << std::endl;
            }
        }

        template <typename U>
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
                std::shared_ptr<T> data(new T[size]);
                file.read(reinterpret_cast<char*>(data.get()), size * sizeof(T));

                file.close();

                return Tensor<T>(data, size, shape);
            } else {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return Tensor<T>();
            }
        }





     private:
        static size_t printTensor(std::ostream& os, const Tensor<T>& tensor, size_t level, size_t offset = 0,size_t block_nums=1) {
            size_t dimSize = tensor.shape[level];
            for (size_t i = 0; i < dimSize; ++i) {
                if (level < tensor.dims - 1) {
                    os << "[";
                    block_nums++;
                    block_nums=printTensor(os, tensor, level + 1, offset + i * tensor.strides[level],block_nums);
                    block_nums--;
                    os << "]";
                    if (i < dimSize - 1) {
                        os << ", ";
                        os << std::endl;
                        for(int l=0;l<block_nums;l++){
                            os<<" ";
                        }
                    }
                } else {
                    os << tensor.data.get()[i + offset];
                    if (i < dimSize - 1) {
                        os << ", ";
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
                        unorderedMap[resultIndex] = tensor.data.get()[i] > unorderedMap[resultIndex] ? tensor.data.get()[i] : unorderedMap[resultIndex];
                    }else{
                        unorderedMap[resultIndex] = tensor.data.get()[i] < unorderedMap[resultIndex] ? tensor.data.get()[i] : unorderedMap[resultIndex];
                    }
                }else{
                    unorderedMap[resultIndex]=tensor.data.get()[i];
                }
            }
            for(auto P :unorderedMap){
                result.data.get()[P.first]=P.second;
            }
            return result;
        }
    };


}// namespace ts