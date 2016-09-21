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

double Sigmoid(double input){
  return 1./(1. + exp(-input));
}

double SigmoidPrime(double input){
  return Sigmoid(input) * (1.-Sigmoid(input));
}

double Gaussian(double input){
  return 1./(sqrt(2*PI) * SIGMA) * exp(-pow((input-MU), 2)/pow(2.*SIGMA, 2));
}

double GaussianPrime(double input){
  return (2. * MU-input)/SIGMA * Gaussian(input);
}

double Tanh(double input){
  return (1+tanh(input))/2;
}

double TanhPrime(double input){
  return 1-pow(tanh(input), 2);
}

#endif //NEURALNETWORKS_ACTIVATIONS_HPP
