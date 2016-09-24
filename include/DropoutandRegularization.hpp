//
// Created by Aman LaChapelle on 9/20/16.
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
