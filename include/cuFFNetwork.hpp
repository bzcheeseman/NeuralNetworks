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

static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
  return (nominator + denominator - 1) / denominator;
}

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
// << CurandGetErrorString(status);

struct cuLayer{

  int gpuid;

  int in, out;

  cudnnTensorDescriptor_t weight;
  Eigen::MatrixXf w; //careful - column-major
  float *dev_w;

  cudnnTensorDescriptor_t bias;
  Eigen::VectorXf b;
  float *dev_b;

  cudnnTensorDescriptor_t zs;
  Eigen::VectorXf z;
  float *dev_z;

  cudnnActivationDescriptor_t activation;
  cudnnTensorDescriptor_t as;
  Eigen::VectorXf a;
  float *dev_a;

  cuLayer(int in, int out, int gpuid);
  ~cuLayer();

  void initTensors(int batchSize);

  void setActivation(cudnnActivationMode_t cudnnActivationFunc);

  void copy_to_device();

  void copy_from_device();

  void free_device_ptr();

  //! CHECK COMPUTATIONS - NOT TOTALLY CONVINCED THEY'RE RIGHT - esp. cublas
  //! might not matter though - as long as it happens the same way every time maybe it just doesn't matter?
  void feedThroughLayer(float *device_ptr_input, int len, int batchSize, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);

  friend std::ostream &operator<<(std::ostream &out, cuLayer &layer);

};

class cuFFNetwork {

  int gpuid;
  int batchSize;

  enum {ReLU, Sigmoid, Tanh} activation_func ;

  cuLayer& hidden_layer;
  cuLayer& output_layer;

  cublasHandle_t cublasHandle;
  cudnnHandle_t cudnnHandle;

  cudnnTensorDescriptor_t input_data;

public:
  cuFFNetwork(int gpuid, int batchSize, cuLayer &hidden, cuLayer &outputs);
  ~cuFFNetwork();

  Eigen::VectorXf feedForward(float *data);

};


#endif //NEURALNETWORKS_CUFFNETWORK_HPP
