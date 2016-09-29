//
// Created by Aman LaChapelle on 9/21/16.
//
// NeuralNetworks
// Copyright (C) 2016  Aman LaChapelle
//
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

#ifndef NEURALNETWORKS_CUFFNETWORK_HPP
#define NEURALNETWORKS_CUFFNETWORK_HPP

#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <chrono>
#include <gflags/gflags.h>
#include <Eigen/Dense>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

#include "cudaKernels.hpp"

/*
 * Alright, time for a CUDNN implementation of what I have already!
 */

#define BW 128

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCurandErrors(status) do{                                  \
    std::stringstream _error;                                          \
    if (status != CURAND_STATUS_SUCCESS){                              \
      _error << "CURAND failure " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
} while (0)

struct cuFFLayer{

  int gpuid;

  int in, out;

  int batchSize;

  /*
   * n = batchSize
   * c = number of outputs from that tensor (feature maps ~ map input into output) - properly only need one
   *      tensor per layer.
   * h = height of feature maps = 1 for fully connected layer (nodes don't depend on each other)
   * w = width of feature maps = 1 for fully connected layer (nodes don't depend on each other)
   */

  Eigen::MatrixXf w;
  float *dev_w;

  Eigen::VectorXf b;
  float *dev_b;

  Eigen::VectorXf z;
  float *dev_z;

  cudnnActivationDescriptor_t activation;
  Eigen::VectorXf a;
  float *dev_a;

  //this is the gradient w.r.t. the previous layer (!)
  cudnnTensorDescriptor_t gradientTensor;
  Eigen::MatrixXf gradient;
  float *dev_gradient;
  Eigen::MatrixXf dw;
  float *dCdw;
  Eigen::VectorXf db;
  float *dCdb;

  cudnnTensorDescriptor_t layerTensor; //apply activation within the layer which apparently works well.

  cuFFLayer(int in, int out, int gpuid, int batchSize);
  ~cuFFLayer();

  void initTensor();

  void setActivation(cudnnActivationMode_t cudnnActivationFunc);

  // keep these separate - want to leave all the info on the gpu during the backprop process (feedforward, gradients, etc.)
  // only copy them back for updates (maybe even do updates on the device and copy it back when the training is done?)
  void copy_to_device();

  void copy_from_device();

  void free_device_ptr();

  //! CHECK COMPUTATIONS - NOT TOTALLY CONVINCED THEY'RE RIGHT - specifically cublas (although the dimensions are right
  //! and nothing explodes...)
  //! Also turns out cublas is column-major too so I'm good - just check that cudnn is too - thougth it might not matter either
  //! might not matter though - as long as it happens the same way every time maybe it just doesn't matter?
  void feedThroughLayer(float *device_ptr_input, int len, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);

  void init_gradient();

  void copy_back_gradient();

  friend std::ostream &operator<<(std::ostream &out, cuFFLayer &layer);

};

class cuFFNetwork {

  int gpuid;
  int batchSize;

  enum {ReLU, Sigmoid, Tanh} activation_func ;

  cuFFLayer& hidden_layer;
  cuFFLayer& output_layer;

  cublasHandle_t cublasHandle;
  cudnnHandle_t cudnnHandle;

  cudnnTensorDescriptor_t input_data;

public:
  cuFFNetwork(int gpuid, int batchSize, cuFFLayer &hidden, cuFFLayer &outputs);
  ~cuFFNetwork();

  Eigen::VectorXf feedForward(float *data);

  double backPropagate(float *correct_out);

};


#endif //NEURALNETWORKS_CUFFNETWORK_HPP
