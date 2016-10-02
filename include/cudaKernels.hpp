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


#ifndef NEURALNETWORKS_CUDAKERNELS_HPP
#define NEURALNETWORKS_CUDAKERNELS_HPP

#include <cuda.h>
#include <cuda_runtime.h>

/*
 * Gotta be careful how you init - make sure however many threads run this in parallel will get this too (do the roundUp thing)
 */

__global__ void fillOnes(float *vec, int size);

__global__ void fillZeros(float *vec, int size);

__global__ void costFunc(float *vec, int size, int batchSize, float *diff);

__global__ void copyVec(float *vec, float *other, int size);
#endif //NEURALNETWORKS_CUDAKERNELS_HPP
