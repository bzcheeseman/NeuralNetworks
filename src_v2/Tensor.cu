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

#include "../include_v2/Tensor.hpp"

Tensor::Tensor(int N, int C, int H, int W, unsigned int gpuid) : gpuid(gpuid), N(N), C(C), H(H), W(W) {

  cpu_data = Eigen::MatrixXf::Zero(N, C);

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc)); // init tensor for this layer

  checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
}

Tensor::Tensor(float *data, int N, int C, int H, int W, unsigned int gpuid) : gpuid(gpuid), N(N), C(C), H(H), W(W) {

  cpu_data = Eigen::MatrixXf(N, C);

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc)); // init tensor for this layer

  checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

  cpu_data = Eigen::Map<Eigen::MatrixXf>(data, N, C); //init cpu_data
  device_data = thrust::device_vector<float>(data, data + N*C); //init device_data;
  raw_device_data = thrust::raw_pointer_cast(&device_data[0]); //init the raw pointer to hand off to cublas/cudnn routines
}

void Tensor::setDeviceData(float *dev_data) {
  device_data = thrust::device_vector<float>(dev_data, dev_data + N*C); //init device_data;
  raw_device_data = thrust::raw_pointer_cast(&device_data[0]); //init the raw pointer to hand off to cublas/cudnn routines
}

void Tensor::setData(float *data) {
  cpu_data = Eigen::Map<Eigen::MatrixXf>(data, N, C);
  device_data = thrust::device_vector<float>(data, data + N*C); //init device_data;
  raw_device_data = thrust::raw_pointer_cast(&device_data[0]); //init the raw pointer to hand off to cublas/cudnn routines
}

void Tensor::copy_back() {
  checkCudaErrors(cudaMemcpyAsync(cpu_data.data(), raw_device_data, N*C*sizeof(float), cudaMemcpyDeviceToHost));
}

