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


#include "../include_v2/Layer.hpp"

FFLayer::FFLayer(int in, int out, unsigned gpuid, int batchSize) :
        inputs(in), outputs(out), gpuid(gpuid), batchSize(batchSize) {

  checkCudaErrors(cublasCreate(&cublasHandle));
  checkCUDNN(cudnnCreate(&cudnnHandle));

  cpu_w = Eigen::MatrixXf(outputs, inputs);
  device_w = thrust::device_vector<float>(outputs*inputs);
  raw_device_w = thrust::raw_pointer_cast(&(device_w[0]));

  cpu_b = Eigen::VectorXf(out);
  device_b = thrust::device_vector<float>(outputs);
  raw_device_b = thrust::raw_pointer_cast(&(device_b[0]));

  ones = thrust::device_vector<float>(batchSize, 1.0f);
  raw_ones = thrust::raw_pointer_cast(&ones[0]);

  cpu_z = Eigen::MatrixXf(outputs, batchSize);
  device_z = thrust::device_vector<float>(outputs*batchSize);
  raw_device_z = thrust::raw_pointer_cast(&(device_z[0]));

  cpu_a = Eigen::MatrixXf(outputs, batchSize);
  device_a = thrust::device_vector<float>(outputs*batchSize);
  raw_device_a = thrust::raw_pointer_cast(&(device_a[0]));

  checkCudaErrors(cudaSetDevice(gpuid));

  checkCUDNN(cudnnCreateActivationDescriptor(&(layerActivation)));
  checkCUDNN(cudnnSetActivationDescriptor(layerActivation, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

  float mean = (float)0.0;
  float stddev = (float)(1.0 /sqrt( (float)in ));

  curandGenerator_t gen; //create generator
  checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); //set generator seed

  auto now = std::chrono::high_resolution_clock::now();
  std::uint64_t nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, nanos)); //set seed from chrono::now()

  checkCurandErrors(curandGenerateNormal(gen, raw_device_w, (std::size_t)(outputs*inputs + (outputs*inputs)%2), mean, stddev)); //generate numbers
  checkCudaErrors(cudaMemcpyAsync(cpu_w.data(), raw_device_w, (outputs*inputs)*sizeof(float), cudaMemcpyDeviceToHost)); //copy it back

  checkCurandErrors(curandGenerateNormal(gen, raw_device_b, (std::size_t)(outputs + outputs%2), mean, stddev)); //generate numbers
  checkCudaErrors(cudaMemcpyAsync(cpu_b.data(), raw_device_b, (outputs)*sizeof(float), cudaMemcpyDeviceToHost)); //copy it back
}

Tensor FFLayer::feedThroughLayer(Tensor &in) {

  checkCudaErrors(cudaSetDevice(gpuid));

  //variables for mixing
  float one = 1.0f, zero = 0.0f;

  //init output tensor
  Tensor out (batchSize, outputs, 1, 1, gpuid);

  //Multiply by our w vector
  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              outputs, inputs, batchSize,
                              &one, raw_device_w, outputs, in.raw_device_data, inputs,
                              &zero, raw_device_z, outputs));

  //Add bias
  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              outputs, batchSize, 1,
                              &one, raw_device_b, outputs, raw_ones, 1,
                              &one, raw_device_z, outputs));

  //Activate - we're transforming the tensor to the Cout size in previous steps so we use that tensor descriptor
  checkCUDNN(cudnnActivationForward(cudnnHandle, layerActivation,
                                    &one, out.TensorDesc, raw_device_z,
                                    &zero, out.TensorDesc, raw_device_a)); //apply activation within the layer

  out.setDeviceData(raw_device_a);

  return out;
}

void FFLayer::initBackprop() {
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

void FFLayer::backThroughLayer(Tensor &backward) {

  initBackprop();

  checkCudaErrors(cudaSetDevice(gpuid));

  //variables for mixing
  float one = 1.0f, zero = 0.0f;

  thrust::device_vector<float> device_cost (batchSize*outputs, 0.0f);
  float *raw_device_cost = thrust::raw_pointer_cast(&device_cost[0]);

  //Cin == the output of next layer (going backwards)

  //Get in the delta from previous layer and its weights
  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, outputs, inputs, batchSize,
                              &one, next->raw_device_w, outputs, backward.raw_device_data, inputs,
                              &zero, raw_device_cost, outputs));

  //Now we activate backward, store it all in delta
  checkCUDNN(cudnnActivationBackward(cudnnHandle, layerActivation,
                                     &one, backward.TensorDesc, this->raw_device_a,
                                     backward.TensorDesc, raw_device_cost,
                                     backward.TensorDesc, this->raw_device_z,
                                     &zero, backward.TensorDesc, this->raw_device_delta));

  //compute bias gradient (collapse along one axis)
  checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, outputs, batchSize,
                              &one, this->raw_device_delta, outputs, raw_ones, 1,
                              &zero, this->raw_device_db, 1));

  //compute weights gradient
  checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, outputs, inputs, batchSize,
                              &one, this->raw_device_delta, outputs, prev->raw_device_a, inputs,
                              &zero, this->raw_device_dw, outputs));

}
