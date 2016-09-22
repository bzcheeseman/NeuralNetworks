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

/*
 * TODO: Make the reader work for arbitrary length
 */

/**
 * @file include/dataReader.hpp
 * @brief Declares methods for reading and storing network training data.
 */

/**
 * @struct dataSet include/dataReader.hpp
 * @brief Holds the data for neural net training, etc.
 *
 * Holds data for training a neural net (or other ML algorithm) in the form of either a T** or an array of
 * Eigen matrices.
 */
template<typename T>
struct dataSet{

  /**
   * Constructor - Makes a new dataSet that can be assigned to.  Malloc's space, etc.
   *
   * @param len Length of the dataset (number of matrices/vectors)
   * @param in Number of inputs (input nodes on the network)
   * @param out Number of outputs (output nodes on the network)
   * @return A new dataSet<T> object
   */
  dataSet(unsigned long len, unsigned long in, unsigned long out): count(len) {
    ins = new T* [len];
    outs = new T* [len];

    inputs = new Eigen::Matrix<T, Eigen::Dynamic, 1>[len];
    outputs = new Eigen::Matrix<T, Eigen::Dynamic, 1>[len];

    for (long i = 0; i < len; i++){
      ins[i] = new T [in];
      outs[i] = new T [out];
    }
  }

  //! Number of data entries
  long count;

  //! Array of arrays (array of input vectors)
  T** ins;
  //! Array of arrays (array of output vectors)
  T** outs;
  //! Array of input vectors
  Eigen::Matrix<T, Eigen::Dynamic, 1>* inputs;
  //! Array of output vectors
  Eigen::Matrix<T, Eigen::Dynamic, 1>* outputs;

};

/**
 * @class dataReader include/dataReader.hpp src/dataReader.cpp
 * @brief Reads data from a file into a dataSet construct
 *
 * Currently only works for the iris dataset and that format - need to make it work for other file formats as well.
 */
class dataReader {
public:
  //! The data that is read in from the file
  dataSet<double>* data;

  /**
   * Reads a file in from the name given.  Currently only works for the iris datasets in the data folder of this project.
   *
   * @param dataset The name of the dataset
   * @param in Number of inputs (could read this from a file)
   * @param out Number of outputs (could read this from a file)
   * @return A new dataReader object that contains the data from a file given by filename.
   */
  dataReader(std::string dataset, long in, long out);

  /**
   * Destructor - default.
   */
  virtual ~dataReader();
};


#endif //MULTI_NODE_NN_DATAREADER_HPP
