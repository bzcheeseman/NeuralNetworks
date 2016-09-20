//
// Created by Aman LaChapelle on 9/18/16.
//

#include <iostream>
#include <string>
#include <math.h>
#include <ctime>
#include "include/FFNetwork.hpp"
#include "include/dataReader.hpp"
#include "include/Activations.hpp"
#include "include/CostFunctions.hpp"

double truncate(double in){
  if (in >= 0.5){
    return 1.;
  }
  else{
    return 0.;
  }
}

int main(int argc, char *argv[]){
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::clock_t start;
  double duration;

  std::vector<unsigned> topo = {4, 3, 3};
  double eta = 5e-2;
  double l = 1e6; //this appears in a denominator - regularization parameter
  double gamma = 0.95;
  double epsilon = 1e-6;
  std::string backprop = "ADADELTA";

  dataReader *train = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_training.dat", 4, 3);
  dataReader *validate = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_validation.dat", 4, 3);
  dataReader *test = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_test.dat", 4, 3);

  FFNetwork *net = new FFNetwork(topo, eta, l, gamma, epsilon);
  net->setFunctions(Tanh, TanhPrime, Identity, QuadCost, QuadCostPrime);
  net->setBackpropAlgorithm(backprop.c_str());
  int corr = 0;
  for (int i = 0; i < test->data->count; i++){
    Eigen::VectorXd out = net->feedForward(test->data->inputs[i]);
    out = out.unaryExpr(&truncate);
    if (out == test->data->outputs[i]){
      corr++;
    }
  }

  std::cout << "Before Training: " << corr << "/" << validate->data->count << " correct" << std::endl;

  int len = train->data->count;

  double goal = 5e-3;
  long max_epochs = 1e9;
  double min_gradient = 5e-3;

  start = std::clock();
  net->Train(train->data, validate->data, goal, max_epochs, min_gradient);
  duration = ( std::clock() - start ) / ((double) CLOCKS_PER_SEC);

  std::cout << "Training took " << duration << " sec" << std::endl << std::endl;


  corr = 0;
  for (int i = 0; i < test->data->count; i++){
    Eigen::VectorXd out = net->feedForward(test->data->inputs[i]);
    out = out.unaryExpr(&truncate);
    if (out == test->data->outputs[i]){
      corr++;
    }
  }

  std::cout << "After Training: " << corr << "/" << test->data->count << " correct" << std::endl << std::endl;

  std::cout << "Raw network output on test dataset:" << std::endl;
  std::cout << net->feedForward(test->data->inputs[0]) << std::endl << std::endl;
  std::cout << "Truncated network output (>=0.5 = 1, <0.5 = 0):" << std::endl;
  std::cout << net->feedForward(test->data->inputs[0]).unaryExpr(&truncate) << std::endl << std::endl;
  std::cout << "Corresponding correct output:" << std::endl;
  std::cout << test->data->outputs[0] << std::endl;



  return 0;
}