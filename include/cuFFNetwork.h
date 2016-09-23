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
  int in, out;

  cudnnTensorDescriptor_t weight;
  Eigen::MatrixXf w; //careful - column-major

  cudnnTensorDescriptor_t bias;
  Eigen::VectorXf b;

  cudnnTensorDescriptor_t zs;
  Eigen::VectorXf z;

  cudnnActivationDescriptor_t activation;
  Eigen::VectorXf a;



  cuLayer(int in, int out): in(in), out(out) {
    w = Eigen::MatrixXf(out, in);
    b = Eigen::VectorXf(out);
    z = Eigen::VectorXf(out);
    a = Eigen::VectorXf(out);

    checkCudaErrors(cudaSetDevice(0));

    float *devicedata;
    float mean = (float)0.0;
    float stddev = (float)(1.0/(float)sqrt( (float)in ));

    curandGenerator_t gen;
    //create generator
    checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    //set generator seed
    auto now = std::chrono::high_resolution_clock::now();
    std::uint64_t nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, nanos)); //set seed here

    checkCudaErrors(cudaMalloc((void **)&devicedata, (in*out)*sizeof(float))); //malloc size of weights
    checkCurandErrors(curandGenerateNormal(gen, devicedata, (std::size_t)(in*out + (in*out)%2), mean, stddev)); //generate numbers
    checkCudaErrors(cudaMemcpy(w.data(), devicedata, (in*out)*sizeof(float), cudaMemcpyDeviceToHost)); //copy it back
    checkCudaErrors(cudaFree(devicedata)); //free pointer to realloc

    checkCudaErrors(cudaMalloc((void **)&devicedata, (out)*sizeof(float))); //realloc for biases
    checkCurandErrors(curandGenerateNormal(gen, devicedata, (std::size_t)(out + out%2), mean, stddev)); //generate numbers
    checkCudaErrors(cudaMemcpy(b.data(), devicedata, (out)*sizeof(float), cudaMemcpyDeviceToHost)); //copy it back

    checkCudaErrors(cudaFree(devicedata)); //free pointer
    checkCurandErrors(curandDestroyGenerator(gen));

    fillZeros<<<BW, BW>>>(z.data(), out); //fill with zeros so we know what's there
    fillZeros<<<BW, BW>>>(a.data(), out); //fill with zeros so we know what's there
    checkCudaErrors(cudaDeviceSynchronize());
  }

  friend std::ostream &operator<<(std::ostream &out, cuLayer &layer){
    out << "Inputs: " << layer.in << " Outputs: " << layer.out << std::endl;
    out << "==========Weights==========\n" << layer.w << std::endl;
    out << "\n==========Bias==========\n"<< layer.b << std::endl;
    out << "\n==========Z==========\n"<< layer.z << std::endl;
    out << "\n==========Activations==========\n"<< layer.a << std::endl;
    return out;
  }


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
  cudnnTensorDescriptor_t hiddenTensor;
  cudnnTensorDescriptor_t outputTensor;

  cudnnActivationDescriptor_t hidden_activations;
  cudnnActivationDescriptor_t output_activations;

public:
  cuFFNetwork(int gpuid, int batchSize, cuLayer &hidden, cuLayer &outputs);
  ~cuFFNetwork();

//  void feedForward(double *data, double *hidden, double* hiddenact, double *output, double *result,
//                                 double *hiddenWeight, double *hiddenBias, double *outputWeight,
//                                 double *outputBias, double *ones);

//  double backPropagate(double *data, double *correct, double *hidden, double *hiddenact,
//                       double *output, double *result, double *hiddenWeight, double *hiddenBias,
//                       double *outputWeight, double *outputBias, double *dCdW_hidden, double *dCdB_hidden, ) //this is done really stupidly
  //more encapsulation in struct etc. - we can hold ALL of this in a layer struct...

};


#endif //NEURALNETWORKS_CUFFNETWORK_HPP
