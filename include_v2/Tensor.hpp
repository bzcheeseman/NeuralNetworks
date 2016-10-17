//
// Created by Aman LaChapelle on 10/16/16.
//
// NeuralNetworks
// Copyright (c) 2016 Aman LaChapelle
// Full license at NeuralNetworks/LICENSE.txt
//

/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef NEURALNETWORKS_TENSOR_HPP
#define NEURALNETWORKS_TENSOR_HPP

#include <iostream>

#include <Eigen/Dense>
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "utility.hpp"

/**
 * @class Tensor include/Tensor.hpp
 *
 * Templated on the dimensions of the tensor - number of objects and number of data values.
 * The height and width of the feature map is set to one - it will be changed if the tensor passes through a convolution
 * layer.
 *
 * n = batchSize
 * c = number of outputs from that tensor (feature maps ~ map input into output) - properly only need one
 *      tensor per layer.
 * h = height of feature maps = 1 for fully connected layer (nodes don't depend on each other)
 * w = width of feature maps = 1 for fully connected layer (nodes don't depend on each other)
 *
 */
struct Tensor {

  cudnnTensorDescriptor_t TensorDesc; //this will have to be changed by each layer it passes through
  unsigned gpuid; //Identifies the gpu we're working on.
  int N, C, H, W;


  Eigen::MatrixXf cpu_data;
  thrust::device_vector<float> device_data;
  float *raw_device_data; //raw pointer cast for &device_data[0] <- maintain this status!

  /**
   * Creates a blank tensor - nothing inside
   *
   * @param cudnnHandle cudnnHandle_t responsible for CUDNN processes
   * @param gpuid The gpu we're using - defaults to 0.
   * @return An empty Tensor object.
   */
  Tensor(int N, int C, int H, int W, unsigned gpuid = 0): gpuid(gpuid), N(N), C(C), H(H), W(W) {

    cpu_data = Eigen::MatrixXf::Zero(N, C);

    checkCudaErrors(cudaSetDevice(gpuid));

    checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc)); // init tensor for this layer

    checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  }

  /**
   *
   * @param data
   * @param cudnnHandle
   * @param gpuid
   * @return A Tensor object filled with data specified by @param data
   */
  Tensor(float *data, int N, int C, int H, int W, unsigned gpuid = 0): gpuid(gpuid), N(N), C(C), H(H), W(W) {

    cpu_data = Eigen::MatrixXf(N, C);

    checkCudaErrors(cudaSetDevice(gpuid));

    checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc)); // init tensor for this layer

    checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    cpu_data = Eigen::Map<Eigen::MatrixXf>(data, N, C); //init cpu_data
    device_data = thrust::device_vector<float>(data, data + N*C); //init device_data;
    raw_device_data = thrust::raw_pointer_cast(&device_data[0]); //init the raw pointer to hand off to cublas/cudnn routines
  }

  /**
   *
   * @param data What to fill the tensor with
   */
  void setDeviceData(float *dev_data){
    device_data = thrust::device_vector<float>(dev_data, dev_data + N*C); //init device_data;
    raw_device_data = thrust::raw_pointer_cast(&device_data[0]); //init the raw pointer to hand off to cublas/cudnn routines
  }

  /**
   *
   * @param data What to fill the tensor with
   */
  void setData(float *data){
    cpu_data = Eigen::Map<Eigen::MatrixXf>(data, N, C);
    device_data = thrust::device_vector<float>(data, data + N*C); //init device_data;
    raw_device_data = thrust::raw_pointer_cast(&device_data[0]); //init the raw pointer to hand off to cublas/cudnn routines
  }

  void copy_back(){
    checkCudaErrors(cudaMemcpyAsync(cpu_data.data(), raw_device_data, N*C*sizeof(float), cudaMemcpyDeviceToHost));
  }



};


#endif //NEURALNETWORKS_TENSOR_HPP
