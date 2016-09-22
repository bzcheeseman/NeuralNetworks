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

#include "../include/dataReader.hpp"

/**
 * @file include/dataReader.cpp
 * @brief Defines methods for reading and storing network training data.
 */


using namespace std;

DEFINE_bool(debug, false, "Sets the debug variable to enable(disable) extra logging");

dataReader::dataReader(std::string dataset, long in, long out) {


  Eigen::MatrixXd inputs;
  Eigen::MatrixXd outputs;

  long counter = 0;

  ifstream incoming (dataset);
  if (incoming.is_open()) {
    while (incoming.good()) {

      inputs.conservativeResize(counter+1, in);
      outputs.conservativeResize(counter+1, out);

      string line;
      getline(incoming, line);
      float x1, x2, x3, x4, x5, x6, x7;
      stringstream ss(line);
      ss >> x1 >> x2 >> x3 >> x4 >> x5 >> x6 >> x7;

      inputs.row(counter) << x1, x2, x3, x4;
      outputs.row(counter) << x5, x6, x7;
      counter++;

    }
    incoming.close();
  }
  else{
    std::cout << "Dataset not found" << std::endl;
  }

  if (FLAGS_debug){
    std::cerr << "Found inputs:\n" << inputs << "\n" << std::endl;
    std::cerr << "Found outputs:\n" << outputs << "\n" << std::endl;
  }

  data = new dataSet<double>(counter, in, out);

#pragma omp parallel for schedule(guided, 4)
  for (long i = 0; i < counter; i++) {
    for (int j = 0; j < in; j++) {
      data->ins[i][j] = inputs(i, j);
    }
    data->inputs[i] = inputs.row(i);
  }

#pragma omp parallel for schedule(guided, 4)
  for (long i = 0; i < counter; i++) {
    for (int j = 0; j < out; j++) {
      data->outs[i][j] = outputs(i, j);
    }
    data->outputs[i] = outputs.row(i);
  }

  if (FLAGS_debug){
    for (long i = 0; i < counter; i++){
      std::cerr << "Matrix values (by row, input with corresponding output):\n";
      std::cerr << inputs.row(i) << "\n";
      std::cerr << outputs.row(i) << "\n\n";

      std::cerr << "Array values (in with corresponding out):\n";
      for (int j = 0; j < in; j++){
        std::cerr << data->ins[i][j] << " ";
      }
      std::cerr << "\n";
      for (int j = 0; j < out; j++){
        std::cerr << data->outs[i][j] << " ";
      }
      std::cerr << "\n\n";
    }

    std::cerr << std::endl;

  }

}

dataReader::~dataReader() {
  ;
}



