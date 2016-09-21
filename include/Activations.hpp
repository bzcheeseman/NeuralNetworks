//
// Created by Aman LaChapelle on 9/19/16.
//

#ifndef NEURALNETWORKS_ACTIVATIONS_HPP
#define NEURALNETWORKS_ACTIVATIONS_HPP

#include <math.h>
#include <Eigen/Dense>

#define PI 3.1415926535897932384
#define SIGMA 1.
#define MU 0.0

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

#endif //NEURALNETWORKS_ACTIVATIONS_HPP
