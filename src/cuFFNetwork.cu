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

#include "../include/cuFFNetwork.hpp"

/*******************************************
 * cuLayer
 *******************************************/

cuLayer::cuLayer(int in, int out, int gpuid) : in(in), out(out), gpuid(gpuid) {
  w = Eigen::MatrixXf(out, in);
  b = Eigen::VectorXf(out);
  z = Eigen::VectorXf(out);
  a = Eigen::VectorXf(out);

  checkCudaErrors(cudaSetDevice(gpuid));

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

  checkCudaErrors(cudaMalloc(&devicedata, (in*out)*sizeof(float))); //malloc size of weights
  checkCurandErrors(curandGenerateNormal(gen, devicedata, (std::size_t)(in*out + (in*out)%2), mean, stddev)); //generate numbers
  checkCudaErrors(cudaMemcpy(w.data(), devicedata, (in*out)*sizeof(float), cudaMemcpyDeviceToHost)); //copy it back
  checkCudaErrors(cudaFree(devicedata)); //free pointer to realloc

  checkCudaErrors(cudaMalloc((void **)&devicedata, (out)*sizeof(float))); //realloc for biases
  checkCurandErrors(curandGenerateNormal(gen, devicedata, (std::size_t)(out + out%2), mean, stddev)); //generate numbers
  checkCudaErrors(cudaMemcpy(b.data(), devicedata, (out)*sizeof(float), cudaMemcpyDeviceToHost)); //copy it back

  float *dev_z, *dev_a;

  checkCudaErrors(cudaMalloc(&dev_z, out * sizeof(float)));
  checkCudaErrors(cudaMemset(dev_z, 0.0f, out*sizeof(float)));
  checkCudaErrors(cudaMemcpy(z.data(), dev_z, out*sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMalloc(&dev_a, out * sizeof(float)));
  checkCudaErrors(cudaMemset(dev_a, 0.0f, out*sizeof(float)));
  checkCudaErrors(cudaMemcpy(a.data(), dev_a, out*sizeof(float), cudaMemcpyDeviceToHost));


  checkCudaErrors(cudaFree(devicedata)); //free pointer
  checkCudaErrors(cudaFree(dev_z));
  checkCudaErrors(cudaFree(dev_a));
  checkCurandErrors(curandDestroyGenerator(gen));

  checkCudaErrors(cudaDeviceSynchronize());
}

cuLayer::~cuLayer() {
  checkCudaErrors(cudaSetDevice(gpuid));
  checkCUDNN(cudnnDestroyTensorDescriptor(weight));
  checkCUDNN(cudnnDestroyTensorDescriptor(bias));
  checkCUDNN(cudnnDestroyTensorDescriptor(zs));
  checkCUDNN(cudnnDestroyTensorDescriptor(as));
  checkCUDNN(cudnnDestroyActivationDescriptor(activation));
}

void cuLayer::initTensors(int batchSize) {

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCUDNN(cudnnCreateTensorDescriptor(&(weight))); // init weight tensor for hidden layer
  checkCUDNN(cudnnCreateTensorDescriptor(&(bias))); // init bias tensor for hidden layer
  checkCUDNN(cudnnCreateTensorDescriptor(&(zs))); // init z tensor for hiddens
  checkCUDNN(cudnnCreateTensorDescriptor(&(as))); // init a tensor for hiddens

  //not sure if this is right...hopefully it is though
  checkCUDNN(cudnnSetTensor4dDescriptor(weight, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(bias, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(zs, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(as, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, out, 1, 1));

}

void cuLayer::setActivation(cudnnActivationMode_t cudnnActivationFunc) {
  checkCudaErrors(cudaSetDevice(gpuid));

  checkCUDNN(cudnnCreateActivationDescriptor(&(activation)));
  checkCUDNN(cudnnSetActivationDescriptor(activation, cudnnActivationFunc, CUDNN_PROPAGATE_NAN, 0.0));
}

void cuLayer::copy_to_device() {

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCudaErrors(cudaMalloc(&dev_w, in*out * sizeof(float)));
  checkCudaErrors(cudaMemcpyAsync(dev_w, &w.data()[0], in*out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_b, out * sizeof(float)));
  checkCudaErrors(cudaMemcpyAsync(dev_b, &b.data()[0], out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_z, out * sizeof(float)));
  checkCudaErrors(cudaMemcpyAsync(dev_z, &z.data()[0], out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_a, out * sizeof(float)));
  checkCudaErrors(cudaMemcpyAsync(dev_a, &a.data()[0], out * sizeof(float), cudaMemcpyHostToDevice));

}

void cuLayer::copy_from_device() {

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCudaErrors(cudaMemcpyAsync(w.data(), dev_w, out*in*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync(b.data(), dev_b, out*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync(z.data(), dev_z, out*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync(a.data(), dev_a, out*sizeof(float), cudaMemcpyDeviceToHost));

}

void cuLayer::free_device_ptr() {

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCudaErrors(cudaFree(dev_w));
  checkCudaErrors(cudaFree(dev_b));
  checkCudaErrors(cudaFree(dev_z));
  checkCudaErrors(cudaFree(dev_a));

}

void cuLayer::feedThroughLayer(float *device_ptr_input, int len, int batchSize, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle) {
  assert(len == in);

  checkCudaErrors(cudaSetDevice(gpuid));

  float one = 1.0f, zero = 0.0f;

  float *ones;
  checkCudaErrors(cudaMalloc(&ones, batchSize * sizeof(float)));
  checkCudaErrors(cudaMemset(ones, 1.0f, batchSize * sizeof(float)));

  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              out, batchSize, out,
                              &one, dev_w, out, device_ptr_input, in,
                              &zero, dev_z, out));

  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              out, batchSize, 1,
                              &one, dev_b, out, ones, 1,
                              &one, dev_z, out));

  checkCUDNN(cudnnActivationForward(cudnnHandle, activation,
                                    &one, zs, dev_z,
                                    &zero, as, dev_a));
}

std::ostream &operator<<(std::ostream &out, cuLayer &layer) {
  out << "Inputs: " << layer.in << " Outputs: " << layer.out << std::endl;
  out << "==========Weights==========\n" << layer.w << std::endl;
  out << "\n==========Bias==========\n"<< layer.b << std::endl;
  out << "\n==========Z==========\n"<< layer.z << std::endl;
  out << "\n==========Activations==========\n"<< layer.a << std::endl;
  return out;
}


/*******************************************
 * cuFFNetwork
 *******************************************/

cuFFNetwork::cuFFNetwork(int gpuid, int batchSize, cuLayer& hidden_layer, cuLayer& output_layer):
        gpuid(gpuid), batchSize(batchSize), hidden_layer(hidden_layer), output_layer(output_layer) {

  //set up device
  checkCudaErrors(cudaSetDevice(gpuid));

  checkCudaErrors(cublasCreate(&cublasHandle));
  checkCUDNN(cudnnCreate(&cudnnHandle));

  activation_func = Tanh;

  checkCUDNN(cudnnCreateTensorDescriptor(&input_data)); // init tensor for input data

  this->hidden_layer.initTensors(batchSize);

  this->output_layer.initTensors(batchSize);

  if (activation_func == ReLU){
    this->hidden_layer.setActivation(CUDNN_ACTIVATION_RELU);
  }
  else if (activation_func == Tanh){
    this->hidden_layer.setActivation(CUDNN_ACTIVATION_TANH);
  }
  else if (activation_func == Sigmoid){
    this->hidden_layer.setActivation(CUDNN_ACTIVATION_SIGMOID);
  }


  this->output_layer.setActivation(CUDNN_ACTIVATION_SIGMOID);

}

cuFFNetwork::~cuFFNetwork() {

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCudaErrors(cublasDestroy(cublasHandle));
  checkCUDNN(cudnnDestroy(cudnnHandle));

  checkCUDNN(cudnnDestroyTensorDescriptor(input_data));

}

Eigen::VectorXf cuFFNetwork::feedForward(float *data) {
  checkCudaErrors(cudaSetDevice(gpuid));

  hidden_layer.copy_to_device();

  output_layer.copy_to_device();

  //Copy data to device to pass to the hidden layer
  float *dev_data;
  checkCudaErrors(cudaMalloc(&dev_data, hidden_layer.in * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_data, &data[0], batchSize*hidden_layer.in * sizeof(float), cudaMemcpyHostToDevice));

  //hidden layer - feed through
  hidden_layer.feedThroughLayer(dev_data, hidden_layer.in, batchSize, cublasHandle, cudnnHandle);

  //output layer - feed through
  output_layer.feedThroughLayer(hidden_layer.dev_a, output_layer.in, batchSize, cublasHandle, cudnnHandle);

  //copy hidden data back to host
  hidden_layer.copy_from_device();
  //free device pointers
  hidden_layer.free_device_ptr();

  //copy output data back to host
  output_layer.copy_from_device();
  //free device pointers
  output_layer.free_device_ptr();

  checkCudaErrors(cudaFree(dev_data));

  checkCudaErrors(cudaDeviceSynchronize());

  return output_layer.a;
}