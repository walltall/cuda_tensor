#include <iostream>
#include <vector>
#include "tensor.cuh"
#include <chrono>
#include "cudaCalculate.cuh"

int main() {

//        double data[5] = {1,2,3,3,4};
//        double data2[5]={1.3,2.4,3,3,4};
//    double data[10][10];
//    double data2[10][10];

        ts::Tensor<double> t=ts::Tensor<double>::rand({8192,8192});
//        std::cout<<t<<std::endl;
        ts::Tensor<double> t2=ts::Tensor<double>::rand({8192,8192});
//    std::cout<<t2<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<3;i++) {
         cu::cuda_mul(t, t2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout  << duration.count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<3;i++) {
       ts::Tensor<double>::mul(t, t2);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout  << duration.count() << " s" << std::endl;

    return 0;
}
//最基础的初始化
//     int data[3][2][2][2] = {{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{{9,
//                             10}, {11, 12}}, {{13, 14}, {15, 16}}}, {{{17,
//                             18}, {19, 20}}, {{21, 22}, {23, 24}}}};
//int data[1][3][2] = {{{1, 2}, {3, 4}, {5, 6}}};
//int data[3][2] = {{1, 2}, {3, 4}, {5, 6}};
//int data[2] = {1, 2};
//    ts::Tensor<int> t(data);

//    std::cout << t << std::endl;

//随机初始化
//ts::Tensor<int> t = ts::Tensor<int>::full({3, 3}, 6);
//    ts::Tensor<int> tt = ts::Tensor<int>::transpose(t, 0, 1);
//    std::cout << t(0, {0, 1}) << std::endl;
//    std::cout << t(1, {0, 1}) << std::endl;

//    double data[1][2][3] = {{{1,2,3},{4,5,6}}};
//    double pp[1][3][2] = {{{1,2},{3,4},{5,6}}};
//    ts::Tensor<double> t(data);
//    ts::Tensor<double> tt(pp);
//    ts::Tensor<double>res=ts::Tensor<double>::bmm(t,tt);

//    ts::Tensor<double> tt(pp);
//    ts::Tensor<double> ans=ts::Tensor<double>::sub(t,tt);
//     std::cout << res << std::endl;

//随机初始化

// test mutation
//    int data2[2] = {2, 1};
//    ts::Tensor<int> ttt(data2);
//    std::cout << ttt << std::endl;
//    std::cout << t(1) << std::endl;
//    t(1) = 1;
//    std::cout << t << std::endl;
//    t(1, {0, 2}) = ttt;
//    std::cout << t << std::endl;

// test transpose & permutation

//    ts::Tensor<int> transpose1 = ts::Tensor<int>::transpose(t, 2, 3);
//    std::cout << transpose1 << std::endl;
//    ts::Tensor<int> transpose2 = t.transpose(0, 1);
//    std::cout << transpose2 << std::endl;
//    std::vector<int> permute1= {1, 0};
//    ts::Tensor<int> permutation1 = ts::Tensor<int>::permute(t, permute1);
//    std::cout << permutation1 << std::endl;
//    ts::Tensor<int> permutation2 = t.permute(permute1);
//    std::cout << permutation2 << std::endl;

// test view
//    std::vector<int> view_shape = {6, 1};
//    ts::Tensor<int> view1 = ts::Tensor<int>::view(t, view_shape);
//    std::cout << view1 << std::endl;
//    ts::Tensor<int> view2 = t.view(view_shape);
//    std::cout << view2 << std::endl;
//
//    ts::Tensor<double>::save<double>(t.mul(t2),"E:\\cpp_tensor_project\\cs205_tensor\\res.txt");
//    ts::Tensor<double>res=ts::Tensor<double>::load<double>("E:\\cpp_tensor_project\\cs205_tensor\\res.txt");
//    double data[3][3] = {
//            {1.08,1.14,-0.17},
//            {0.5,0.88,0.33},
//            {3.2,323, 22}
//    };
//    ts::Tensor<double> t(data);
////        std::cout<<ts::Tensor<double>::log(t)<<std::endl;
//    ts::Tensor<double>res=ts::Tensor<double>::diagonal(t,0);
//    std::cout<<res<<std::endl;