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

#include "../include/cuFFNetwork.h"

cuFFNetwork::cuFFNetwork(int gpuid, int batchSize, cuLayer& hidden_layer, cuLayer& output_layer):
        gpuid(gpuid), hidden_layer(hidden_layer), output_layer(output_layer) {

  //set up device
  checkCudaErrors(cudaSetDevice(gpuid));
  checkCudaErrors(cublasCreate(&cublasHandle));
  checkCUDNN(cudnnCreate(&cudnnHandle));

  checkCUDNN(cudnnCreateTensorDescriptor(&input_data)); // init tensor for input data
  checkCUDNN(cudnnCreateTensorDescriptor(&hiddenTensor)); // init tensor for hidden layer
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor)); // init tensor for output layer

  checkCUDNN(cudnnCreateActivationDescriptor(&hidden_activations)); // init hidden layer activations
  checkCUDNN(cudnnCreateActivationDescriptor(&output_activations)); // init output layer activations

  checkCUDNN(cudnnSetTensor4dDescriptor(hiddenTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batchSize, hidden_layer.out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batchSize, output_layer.out, 1, 1));

  if (activation_func == ReLU){
    checkCUDNN(cudnnSetActivationDescriptor(hidden_activations, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
  }
  else if (activation_func == Tanh){
    checkCUDNN(cudnnSetActivationDescriptor(hidden_activations, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));
  }
  else if (activation_func == Sigmoid){
    checkCUDNN(cudnnSetActivationDescriptor(hidden_activations, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));
  }


  checkCUDNN(cudnnSetActivationDescriptor(output_activations, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

}

cuFFNetwork::~cuFFNetwork() {

  checkCudaErrors(cudaSetDevice(gpuid));
  checkCudaErrors(cublasDestroy(cublasHandle));
  checkCUDNN(cudnnDestroy(cudnnHandle));

  checkCUDNN(cudnnDestroyTensorDescriptor(input_data));
  checkCUDNN(cudnnDestroyTensorDescriptor(hiddenTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));

  checkCUDNN(cudnnDestroyActivationDescriptor(hidden_activations));
  checkCUDNN(cudnnDestroyActivationDescriptor(output_activations));

}

//void cuFFNetwork::feedForward(double *data, double *hidden, double* hiddenact,
//                                            double *output, double *result,
//                                            double *hiddenWeight, double *hiddenBias,
//                                            double *outputWeight, double *outputBias,
//                              double *ones) {
//  double alpha = 1.0;
//  double beta = 0.0;
//
//  checkCudaErrors(cudaSetDevice(gpuid));
//
//  /*
//   * C = alpha * op(A) * op(B) + beta*C
//   *
//   * cublasSgemm(handler, op1, op2, rows of op(A) = rows of C, cols of op(B) = cols of C, cols of op(A) = rows of op(B), scalar alpha,
//   *             input array A, leading dimension of A (rows of A if op(A) = A, cols of A if op(A) = A.transpose() or A.dagger(),
//   *             input array B, leading dimension of A (rows of B if op(B) = B, cols of B if op(B) = B.transpose() or B.dagger(),
//   *             scalar beta, input array C dims = rows of op(A) x cols of op(B), leading dimension of C (rows of op(A))
//   */
//
//  //Hidden layer forward propagation
//  //weights
//
//
//}
//

