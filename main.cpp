//
// Created by Aman LaChapelle on 9/18/16.
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

#include <iostream>
#include <string>
#include <math.h>
#include <ctime>
#include "include/FFNetwork.hpp"
#include "include/dataReader.hpp"
#include "include/Activations.hpp"
#include "include/CostFunctions.hpp"

using namespace std;

int main(int argc, char *argv[]){
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<unsigned> topo = {4, 4, 3, 3};
  std::vector<unsigned> dropout = {0, 1, 1, 0};
  double eta = 5e-2;
  double l = 5e4; //this appears in a denominator - regularization parameter
  double gamma = 0.9;
  double epsilon = 1e-6;
  std::string backprop = "ADADELTA"; // AdaDelta seems to like the quadratic cost best, and is not happy with ReLU probably because gradients
                                   // are ill-formed (ish) - it's weird

  dataReader *train = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_training.dat", 4, 3);
  dataReader *validate = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_validation.dat", 4, 3);
  dataReader *test = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_test.dat", 4, 3);

  FFNetwork *net = new FFNetwork(topo, dropout, eta, l, gamma, epsilon);
  net->setFunctions(Sigmoid, SigmoidPrime, Identity, Bernoulli, truncate, QuadCost, QuadCostPrime);
  net->setBackpropAlgorithm(backprop.c_str());
  int corr = 0;
  for (int i = 0; i < test->data->count; i++){
    Eigen::VectorXi out = (*net)(test->data->inputs[i]);
    if (out == test->data->outputs[i].cast<int>()){
      corr++;
    }
  }

  std::cout << "Before Training: " << corr << "/" << validate->data->count << " correct" << std::endl;

  int len = train->data->count;

  double goal = 1e-4;
  long max_epochs = 1e9;
  double min_gradient = 1e-5;


  net->Train(train->data, validate->data, goal, max_epochs, min_gradient);


  corr = 0;
  for (int i = 0; i < test->data->count; i++){
    Eigen::VectorXi out = (*net)(test->data->inputs[i]);
    if (out == test->data->outputs[i].cast<int>()){
      corr++;
    }
  }

  std::cout << "After Training: " << corr << "/" << test->data->count << " correct" << std::endl << std::endl;

  std::cout << "Raw network output on test dataset:" << std::endl;
  std::cout << net->feedForward(test->data->inputs[0]) << std::endl << std::endl;
  std::cout << "Truncated network output:" << std::endl;
  std::cout << (*net)(test->data->inputs[0]) << std::endl << std::endl;
  std::cout << "Corresponding correct output:" << std::endl;
  std::cout << test->data->outputs[0] << std::endl << std::endl;

//  std::cout << *net << std::endl;

  std::ofstream net_out("/users/aman/code/NeuralNetworks/logging/netTest.log");
  net_out << *net;
  net_out.close();

  return 0;
}