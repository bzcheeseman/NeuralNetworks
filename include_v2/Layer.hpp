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
#include <curand.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


#include "utility.hpp"
#include "Tensor.hpp"

class Layer {

public:
  thrust::device_vector<float> ones; //size = batchSize
  float *raw_ones;

  float *raw_device_w;

  float *raw_device_b;

  float *raw_device_dw;

  float *raw_device_db;

  float *raw_device_delta;

  float *raw_device_z;

  float *raw_device_a;

  Layer *next;
  Layer *prev;

public:
  virtual Tensor feedThroughLayer(Tensor &in) = 0;
  virtual void initBackprop() = 0;
  virtual void backThroughLayer(Tensor &backward) = 0;
//  virtual void update() = 0;

};

class FFLayer : public Layer {

protected:
  unsigned gpuid;
  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;
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
  FFLayer(int in, int out, unsigned gpuid, int batchSize);

  /**
   *
   * @param in
   * @return
   */
  Tensor feedThroughLayer(Tensor &in);

  /**
   *
   */
  void initBackprop();

  /**
   * This one is tricky - gotta check it and make sure it works ok
   *
   * next Holds: the delta that came backwards from the previous layer
   * next Shape: Analogous to what layerTensor would be for this layer - the shape of this layer.
   *
   * @param next The layer closer to the output end of the NN
   * @param prev The layer closer to the input end of the NN
   */
  void backThroughLayer(Tensor &backward);


};



#endif //NEURALNETWORKS_LAYER_HPP
