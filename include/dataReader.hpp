//
// Created by Aman LaChapelle on 6/10/16.
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

#ifndef MULTI_NODE_NN_DATAREADER_HPP
#define MULTI_NODE_NN_DATAREADER_HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <gflags/gflags.h>
#include <omp.h>

DECLARE_bool(debug);

template<typename T>
struct dataSet{

  dataSet(unsigned long len, unsigned long in, unsigned long out): count(len) {
    ins = new T* [len];
    outs = new T* [len];

    inputs = new Eigen::Matrix<T, Eigen::Dynamic, 1>[len];
    outputs = new Eigen::Matrix<T, Eigen::Dynamic, 1>[len];

    for (long i = 0; i < len; i++){
      ins[i] = new T [in];
      outs[i] = new T [out];
//      inputs[i] = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(in);
//      outputs[i] = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(out);
    }
  }

  long count;

  T** ins;
  T** outs;
  Eigen::Matrix<T, Eigen::Dynamic, 1>* inputs;
  Eigen::Matrix<T, Eigen::Dynamic, 1>* outputs;

};

class dataReader {
public:
  dataSet<double>* data;

  dataReader(std::string dataset, long in, long out);
  virtual ~dataReader();

};


#endif //MULTI_NODE_NN_DATAREADER_HPP
