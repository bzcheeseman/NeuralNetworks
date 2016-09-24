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

static inline unsigned int RoundUp(unsigned int numerator, unsigned int denominator)
{
  return (numerator + denominator - 1) / denominator;
}


/*******************************************
 * cuLayer
 *******************************************/

cuFFLayer::cuFFLayer(int in, int out, int gpuid) : in(in), out(out), gpuid(gpuid) {
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

cuFFLayer::~cuFFLayer() {
  checkCudaErrors(cudaSetDevice(gpuid));
  checkCUDNN(cudnnDestroyTensorDescriptor(layerTensor));
  checkCUDNN(cudnnDestroyActivationDescriptor(activation));
}

void cuFFLayer::initTensor(int batchSize) {

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCUDNN(cudnnCreateTensorDescriptor(&(layerTensor))); // init tensor for this layer

  checkCUDNN(cudnnSetTensor4dDescriptor(layerTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, out, 1, 1));

}

void cuFFLayer::setActivation(cudnnActivationMode_t cudnnActivationFunc) {
  checkCudaErrors(cudaSetDevice(gpuid));

  checkCUDNN(cudnnCreateActivationDescriptor(&(activation)));
  checkCUDNN(cudnnSetActivationDescriptor(activation, cudnnActivationFunc, CUDNN_PROPAGATE_NAN, 0.0));
}

void cuFFLayer::copy_to_device() {

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

void cuFFLayer::copy_from_device() {

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCudaErrors(cudaMemcpyAsync(w.data(), dev_w, out*in*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync(b.data(), dev_b, out*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync(z.data(), dev_z, out*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync(a.data(), dev_a, out*sizeof(float), cudaMemcpyDeviceToHost));

}

void cuFFLayer::free_device_ptr() {

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCudaErrors(cudaFree(dev_w));
  checkCudaErrors(cudaFree(dev_b));
  checkCudaErrors(cudaFree(dev_z));
  checkCudaErrors(cudaFree(dev_a));

}

void cuFFLayer::feedThroughLayer(float *device_ptr_input, int len, int batchSize, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle) {
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
                                    &one, layerTensor, dev_z,
                                    &zero, layerTensor, dev_a)); //apply activation within the layer, before giving away output
}

std::ostream &operator<<(std::ostream &out, cuFFLayer &layer) {
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

cuFFNetwork::cuFFNetwork(int gpuid, int batchSize, cuFFLayer& hidden_layer, cuFFLayer& output_layer):
        gpuid(gpuid), batchSize(batchSize), hidden_layer(hidden_layer), output_layer(output_layer) {

  //set up device
  checkCudaErrors(cudaSetDevice(gpuid));

  checkCudaErrors(cublasCreate(&cublasHandle));
  checkCUDNN(cudnnCreate(&cudnnHandle));

  activation_func = Tanh;

  checkCUDNN(cudnnCreateTensorDescriptor(&input_data)); // init tensor for input data

  this->hidden_layer.initTensor(batchSize);

  this->output_layer.initTensor(batchSize);

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

double cuFFNetwork::backPropagate(float *correct_out) {

  checkCudaErrors(cudaSetDevice(gpuid));
  float one = 1.0f, zero = 0.0f;

  hidden_layer.copy_to_device();
  output_layer.copy_to_device();

  float *dev_loss;
  checkCudaErrors(cudaMalloc(&dev_loss, output_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpyAsync(dev_loss, output_layer.dev_a, output_layer.out * sizeof(float), cudaMemcpyDeviceToDevice));

  float *dev_correct;
  checkCudaErrors(cudaMalloc(&dev_correct, output_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpyAsync(dev_correct, correct_out, output_layer.out * sizeof(float), cudaMemcpyHostToDevice));

  //compute error at the last layer - need to update this probably
  costFunc<<<RoundUp(batchSize, BW),BW>>>(dev_loss, output_layer.out, batchSize, dev_correct);

  //think about activationBackward function call carefully - it's gotta be simple-ish
//  checkCUDNN(cudnnActivationBackward(cudnnHandle, output_layer.activation,
//                                     &one, output_layer.layerTensor, output_layer.dev_a, output_layer.layerTensor, dev_loss, ))

  Eigen::VectorXf readin (output_layer.out);
  checkCudaErrors(cudaMemcpyAsync(readin.data(), dev_loss, output_layer.out * sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << readin << std::endl;

}
