//
// Created by Aman LaChapelle on 6/10/16.
//

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
DECLARE_string(logging_dir);
DECLARE_string(data_dir);

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
