//
// Created by Aman LaChapelle on 9/20/16.
//

#ifndef NEURALNETWORKS_DROPOUTANDREGULARIZATION_HPP
#define NEURALNETWORKS_DROPOUTANDREGULARIZATION_HPP

#include <random>

#define DROPOUT 0.8

//Regularization
inline double Identity(double input){
  return input;
}

inline double Sign(double input){
  if (input > 0){
    return 1.;
  }
  else if (input < 0){
    return -1.;
  }
  else{
    return 0.;
  }
}

inline double Zero(double input){
  return 0.;
}

//Dropout
inline double Bernoulli(double input){
  std::random_device rand;
  std::mt19937 generator(rand());
  std::binomial_distribution<> dist(1, DROPOUT);

  return (double)dist(generator);
}

//Truncation
inline double truncate(double in){
  if (in >= 0.5){
    return 1.;
  }
  else{
    return 0.;
  }
}

#endif //NEURALNETWORKS_DROPOUTANDREGULARIZATION_HPP
