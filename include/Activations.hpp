//
// Created by Aman LaChapelle on 9/19/16.
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

#ifndef NEURALNETWORKS_ACTIVATIONS_HPP
#define NEURALNETWORKS_ACTIVATIONS_HPP

#include <math.h>
#include <Eigen/Dense>

#define PI 3.1415926535897932384
#define SIGMA 1.
#define MU 0.0

inline double max(double one, double two){
  return one > two ? one : two;
}

inline double Sigmoid(double input){
  return 1./(1. + exp(-input));
}

inline double SigmoidPrime(double input){
  return Sigmoid(input) * (1.-Sigmoid(input));
}

inline double Gaussian(double input){
  return 1./(sqrt(2*PI) * SIGMA) * exp(-pow((input-MU), 2)/pow(2.*SIGMA, 2));
}

inline double GaussianPrime(double input){
  return (2. * MU-input)/SIGMA * Gaussian(input);
}

inline double Tanh(double input){
  return (1+tanh(input))/2;
}

inline double TanhPrime(double input){
  return 1-pow(tanh(input), 2);
}

// NOT working for some reason
inline double ReLU(double input){
  return max(0.0, input);
}

inline double ReLUPrime(double input){
  if (input > 0.0){
    return 1.0;
  }
  else{
    return 1e-3;
  }
}

#endif //NEURALNETWORKS_ACTIVATIONS_HPP
