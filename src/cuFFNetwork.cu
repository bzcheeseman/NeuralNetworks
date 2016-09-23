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

cuLayer::cuLayer(int in, int out) : in(in), out(out) {
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
  checkCUDNN(cudnnCreateTensorDescriptor(&(hidden_layer.weight))); // init weight tensor for hidden layer
  checkCUDNN(cudnnCreateTensorDescriptor(&(hidden_layer.bias))); // init bias tensor for hidden layer
  checkCUDNN(cudnnCreateTensorDescriptor(&(hidden_layer.zs))); // init z tensor for hiddens
  checkCUDNN(cudnnCreateTensorDescriptor(&(hidden_layer.as))); // init a tensor for hiddens

  checkCUDNN(cudnnCreateTensorDescriptor(&(output_layer.weight))); // init weight tensor for output layer
  checkCUDNN(cudnnCreateTensorDescriptor(&(output_layer.bias))); // init bias tensor for output layer
  checkCUDNN(cudnnCreateTensorDescriptor(&(output_layer.zs))); // init z tensor for hiddens
  checkCUDNN(cudnnCreateTensorDescriptor(&(output_layer.as))); // init a tensor for hiddens

  checkCUDNN(cudnnCreateActivationDescriptor(&(hidden_layer.activation))); // init hidden layer activations
  checkCUDNN(cudnnCreateActivationDescriptor(&(output_layer.activation))); // init output layer activations

  checkCUDNN(cudnnSetTensor4dDescriptor(hidden_layer.weight, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, hidden_layer.out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(hidden_layer.bias, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, hidden_layer.out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(hidden_layer.zs, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, hidden_layer.out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(hidden_layer.as, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, hidden_layer.out, 1, 1));

  checkCUDNN(cudnnSetTensor4dDescriptor(output_layer.weight, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, output_layer.out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_layer.bias, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, output_layer.out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_layer.zs, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, output_layer.out, 1, 1));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_layer.as, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, output_layer.out, 1, 1));

  if (activation_func == ReLU){
    checkCUDNN(cudnnSetActivationDescriptor(hidden_layer.activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
  }
  else if (activation_func == Tanh){
    checkCUDNN(cudnnSetActivationDescriptor(hidden_layer.activation, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));
  }
  else if (activation_func == Sigmoid){
    checkCUDNN(cudnnSetActivationDescriptor(hidden_layer.activation, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));
  }


  checkCUDNN(cudnnSetActivationDescriptor(output_layer.activation, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

  checkCUDNN(cudnnCreateOpTensorDescriptor(&mult));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&add));

  checkCUDNN(cudnnSetOpTensorDescriptor(mult, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  checkCUDNN(cudnnSetOpTensorDescriptor(add, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

}

cuFFNetwork::~cuFFNetwork() {

  checkCudaErrors(cudaSetDevice(gpuid));
  checkCudaErrors(cublasDestroy(cublasHandle));
  checkCUDNN(cudnnDestroy(cudnnHandle));

  checkCUDNN(cudnnDestroyTensorDescriptor(input_data));

}

Eigen::VectorXf cuFFNetwork::feedForward(float *data) {
  checkCudaErrors(cudaSetDevice(gpuid));

  float one = 1.0f, zero = 0.0f;

  //need to copy all these vectors into the device - call cudaMemcpy, etc. for the shit to actually happen...
  //use cudaMemcpyHostToDevice to copy everything in

  float *dev_hidden_w, *dev_hidden_b, *dev_hidden_z, *dev_hidden_a;

  //malloc and copy over hidden layer parameters
  checkCudaErrors(cudaMalloc(&dev_hidden_w, hidden_layer.in*hidden_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_hidden_w, &hidden_layer.w.data()[0],
                                  hidden_layer.in*hidden_layer.out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_hidden_b, hidden_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_hidden_b, &hidden_layer.b.data()[0],
                                  hidden_layer.out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_hidden_z, hidden_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_hidden_z, &hidden_layer.z.data()[0],
                                  hidden_layer.out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_hidden_a, hidden_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_hidden_a, &hidden_layer.a.data()[0],
                                  hidden_layer.out * sizeof(float), cudaMemcpyHostToDevice));

  float *dev_output_w, *dev_output_b, *dev_output_z, *dev_output_a;

  //malloc and copy over output layer parameters
  checkCudaErrors(cudaMalloc(&dev_output_w, output_layer.in*output_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_output_w, &output_layer.w.data()[0],
                                  output_layer.in*output_layer.out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_output_b, output_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_output_b, &output_layer.b.data()[0],
                                  output_layer.out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_output_z, output_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_output_z, &output_layer.z.data()[0],
                             output_layer.out * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_output_a, output_layer.out * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_output_a, &output_layer.a.data()[0],
                             output_layer.out * sizeof(float), cudaMemcpyHostToDevice));


  float *dev_data;
  checkCudaErrors(cudaMalloc(&dev_data, hidden_layer.in * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_data, &data[0], batchSize*hidden_layer.in * sizeof(float), cudaMemcpyHostToDevice));


  float *ones;
  checkCudaErrors(cudaMalloc(&ones, batchSize * sizeof(float)));
  checkCudaErrors(cudaMemset(ones, 1.0f, batchSize * sizeof(float)));

  //! CHECK COMPUTATIONS - NOT TOTALLY CONVINCED THEY'RE RIGHT - esp. cublas

  //hidden layer

  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              hidden_layer.out, batchSize, hidden_layer.out,
                              &one, dev_hidden_w, hidden_layer.out, dev_data, hidden_layer.in,
                              &zero, dev_hidden_z, hidden_layer.out));

  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              hidden_layer.out, batchSize, 1,
                              &one, dev_hidden_b, hidden_layer.out, ones, 1,
                              &one, dev_hidden_z, hidden_layer.out));

  checkCUDNN(cudnnActivationForward(cudnnHandle, hidden_layer.activation,
                                    &one, hidden_layer.zs, dev_hidden_z,
                                    &zero, hidden_layer.as, dev_hidden_a));

  //output layer

  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              output_layer.out, batchSize, output_layer.in,
                              &one, dev_output_w, output_layer.out, dev_hidden_a, output_layer.in,
                              &zero, dev_output_z, output_layer.out));

  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              output_layer.out, batchSize, 1,
                              &one, dev_output_b, output_layer.out, ones, 1,
                              &one, dev_output_z, output_layer.out));

  checkCUDNN(cudnnActivationForward(cudnnHandle, output_layer.activation,
                                    &one, output_layer.zs, dev_output_z,
                                    &zero, output_layer.as, dev_output_a));

  //copy hidden data back to host
  checkCudaErrors(cudaMemcpy(hidden_layer.z.data(), dev_hidden_z, hidden_layer.out*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hidden_layer.a.data(), dev_hidden_a, hidden_layer.out*sizeof(float), cudaMemcpyDeviceToHost));

  //copy output data back to host
  checkCudaErrors(cudaMemcpy(output_layer.z.data(), dev_output_z, output_layer.out*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(output_layer.a.data(), dev_output_a, output_layer.out*sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(ones));
  checkCudaErrors(cudaFree(dev_data));
  checkCudaErrors(cudaFree(dev_hidden_w));
  checkCudaErrors(cudaFree(dev_hidden_b));
  checkCudaErrors(cudaFree(dev_hidden_z));
  checkCudaErrors(cudaFree(dev_hidden_a));
  checkCudaErrors(cudaFree(dev_output_w));
  checkCudaErrors(cudaFree(dev_output_b));
  checkCudaErrors(cudaFree(dev_output_z));
  checkCudaErrors(cudaFree(dev_output_a));

  checkCudaErrors(cudaDeviceSynchronize());

  return output_layer.a;
}