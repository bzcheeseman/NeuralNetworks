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


#ifndef NEURALNETWORKS_LAYER_HPP
#define NEURALNETWORKS_LAYER_HPP

#include <iostream>

#include <Eigen/Dense>
#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


#include "utility.hpp"
#include "Tensor.hpp"

template<int N, int Cin, int Cout, int H, int W>
class Layer {

protected:
  unsigned gpuid;
  cudnnHandle_t *cudnnHandle;
  cublasHandle_t *cublasHandle;
  cudnnActivationDescriptor_t layerActivation;
  int inputs;
  int outputs;
  int batchSize;

  thrust::device_vector<float> ones; //size = batchSize
  float *raw_ones;

  Eigen::MatrixXf cpu_w;
  thrust::device_vector<float> device_w;
  float *raw_device_w;

  Eigen::VectorXf cpu_b;
  thrust::device_vector<float> device_b;
  float *raw_device_b;

  Eigen::MatrixXf cpu_dw;
  thrust::device_vector<float> device_dw;
  float *raw_device_dw;

  Eigen::VectorXf cpu_db;
  thrust::device_vector<float> device_db;
  float *raw_device_db;

  Eigen::MatrixXf cpu_delta;
  thrust::device_vector<float> device_delta;
  float *raw_device_delta;

  Eigen::MatrixXf cpu_z;
  thrust::device_vector<float> device_z;
  float *raw_device_z;

  Eigen::MatrixXf cpu_a;
  thrust::device_vector<float> device_a;
  float *raw_device_a;


public:
  virtual Tensor<N, Cout, H, W> feedThroughLayer(Tensor<N, Cin, H, W> &in) = 0;
  virtual Tensor<N, Cin, H, W> backThroughLayer(Tensor<N, Cout, H, W> &prev) = 0;
  virtual void update() = 0;

};

template<int N, int Cin, int Cout>
class FFLayer : public Layer<N, Cin, Cout, 1, 1> {


public:

  /**
   * Initializes the layer.  Will likely have to split into multiple different functions whether we are just feeding
   * forward or if we are planning to train also.
   *
   * @param in
   * @param out
   * @param gpuid
   * @param batchSize
   * @return
   */
  FFLayer(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, int in, int out, unsigned gpuid, int batchSize):
          inputs(in), outputs(out), gpuid(gpuid), cublasHandle(&cublasHandle), cudnnHandle(&cudnnHandle) {

    cpu_w = Eigen::MatrixXf(out, in);
    device_w = thrust::device_vector<float>(out*in);
    raw_device_w = thrust::raw_pointer_cast(&(device_w[0]));

    cpu_b = Eigen::VectorXf(out);
    device_b = thrust::device_vector<float>(out);
    raw_device_b = thrust::raw_pointer_cast(&(device_b[0]));

    ones = thrust::device_vector<float>(batchSize, 1.0f);
    raw_ones = thrust::raw_pointer_cast(&ones[0]);

    cpu_z = Eigen::MatrixXf(outputs, batchSize);
    device_z = thrust::device_vector<float>(out*batchSize);
    raw_device_z = thrust::raw_pointer_cast(&(device_z[0]));

    checkCudaErrors(cudaSetDevice(gpuid));

    float *devicedata;
    float mean = (float)0.0;
    float stddev = (float)(1.0 /sqrt( (float)in ));

    curandGenerator_t gen; //create generator
    checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); //set generator seed

    auto now = std::chrono::high_resolution_clock::now();
    std::uint64_t nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, nanos)); //set seed from chrono::now()

    checkCurandErrors(curandGenerateNormal(gen, raw_device_w, (std::size_t)(out*in + (out*in)%2), mean, stddev)); //generate numbers
    checkCudaErrors(cudaMemcpyAsync(cpu_w.data(), raw_device_w, (out*in)*sizeof(float), cudaMemcpyDeviceToHost)); //copy it back

    checkCurandErrors(curandGenerateNormal(gen, raw_device_b, (std::size_t)(out + out%2), mean, stddev)); //generate numbers
    checkCudaErrors(cudaMemcpyAsync(cpu_b.data(), raw_device_b, (out)*sizeof(float), cudaMemcpyDeviceToHost)); //copy it back
  }

  /**
   *
   * @param in
   * @return
   */
  Tensor<N, Cout, 1, 1> feedThroughLayer(Tensor<N, Cin, 1, 1> &in){

    checkCudaErrors(cudaSetDevice(gpuid));

    //variables for mixing
    float one = 1.0f, zero = 0.0f;

    //init output tensor
    Tensor<N, Cout, 1, 1> out (gpuid);

    //Multiply by our w vector
    checkCudaErrors(cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                outputs, inputs, batchSize,
                                &one, raw_device_w, outputs, in.raw_device_data, inputs,
                                &zero, raw_device_z, outputs));

    //Add bias
    checkCudaErrors(cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                outputs, batchSize, 1,
                                &one, raw_device_b, out, raw_ones, 1,
                                &one, raw_device_z, out));

    //Activate - we're transforming the tensor to the Cout size in previous steps so we use that tensor descriptor
    checkCUDNN(cudnnActivationForward(*cudnnHandle, layerActivation,
                                      &one, out.cudnnDesc, raw_device_z,
                                      &zero, out.cudnnDesc, raw_device_a)); //apply activation within the layer

    out.setData(raw_device_a);

    return out;
  }

  /**
   *
   */
  void initBackprop(){
    cpu_delta = Eigen::MatrixXf(outputs, batchSize);
    device_delta = thrust::device_vector<float>(outputs*batchSize);
    raw_device_delta = thrust::raw_pointer_cast(&(device_delta[0]));

    cpu_dw = Eigen::MatrixXf(outputs, inputs);
    device_dw = thrust::device_vector<float>(outputs*inputs);
    raw_device_dw = thrust::raw_pointer_cast(&(device_w[0]));

    cpu_db = Eigen::VectorXf(outputs);
    device_db = thrust::device_vector<float>(outputs);
    raw_device_db = thrust::raw_pointer_cast(&(device_w[0]));
  }

  /**
   * This one is tricky - gotta check it and make sure it works ok
   *
   * next Holds: the delta that came backwards from the previous layer
   * next Shape: Analogous to what layerTensor would be for this layer - the shape of this layer.
   *
   * @param next The layer closer to the output end of the NN
   * @param prev The layer closer to the input end of the NN
   */
  void backThroughLayer(Tensor<N, Cout, 1, 1> &next, float *next_device_w, float *prev_device_a){

    checkCudaErrors(cudaSetDevice(gpuid));

    //variables for mixing
    float one = 1.0f, zero = 0.0f;

    Tensor<N, Cin, 1, 1> prev (gpuid);

    thrust::device_vector<float> device_cost (batchSize*outputs, 0.0f);
    float *raw_device_cost;

    //Cin == the output of next layer (going backwards)

    //Get in the delta from previous layer and its weights
    checkCudaErrors(cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, outputs, inputs, batchSize,
                                &one, next_device_w, outputs, next.raw_device_data, inputs,
                                &zero, raw_device_cost, outputs));

    //Now we activate backward, store it all in delta
    checkCUDNN(cudnnActivationBackward(*cudnnHandle, layerActivation,
                                       &one, next.cudnnDesc, raw_device_a,
                                       next.cudnnDesc, raw_device_cost,
                                       next.cudnnDesc, raw_device_z,
                                       &zero, next.cudnnDesc, raw_device_delta));

    //compute bias gradient (collapse along one axis)
    checkCudaErrors(cublasSgemv(*cublasHandle, CUBLAS_OP_N, outputs, batchSize,
                                &one, raw_device_delta, outputs, raw_ones, 1,
                                &zero, raw_device_db, 1));

    //compute weights gradient
    checkCudaErrors(cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, outputs, inputs, batchSize,
                                &one, raw_device_delta, outputs, prev_device_a, inputs,
                                &zero, raw_device_dw, outputs));

    prev.raw_device_data = raw_device_delta;


    prev_device_a = this->raw_device_a;
    next_device_w = this->raw_device_w;
    next = prev;

  }


};


#endif //NEURALNETWORKS_LAYER_HPP
