//
// Created by Aman LaChapelle on 9/22/16.
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

#include "../include/cudaKernels.hpp"

__global__ void fillOnes(float *vec, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  vec[idx] = 1.0f;
}

__global__ void fillZeros(float *vec, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  vec[idx] = 0.0f;
}

__global__ void costFunc(float *vec1, int size, int batchSize, float* vec2){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batchSize*size)
    return;

  vec1[idx] *= (vec2[idx] - vec1[idx]);
  vec1[idx] /= (float)batchSize;

}

__global__ void copyVec(float *vec, float *other, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  vec[idx] = other[idx];
}
