//
// Created by Aman LaChapelle on 9/18/16.
//

#include <iostream>
#include <math.h>
#include <ctime>
#include "include/Network.hpp"
#include "include/dataReader.hpp"

double Sigmoid(double input){
  return 1./(1. + exp(-input));
}

double Tanh(double input){
  return tanh(input);
}

double TanhPrime(double input){
  return 1-pow(std::tanh(input), 2);
}

double SigmoidPrime(double input){
  return Sigmoid(input) * (1.-Sigmoid(input));
}

Eigen::VectorXd QuadCost(Eigen::VectorXd out, Eigen::VectorXd correct){
  return 0.5 * (out - correct).array().pow(2);
}

Eigen::VectorXd QuadCostPrime(Eigen::VectorXd out, Eigen::VectorXd correct, Eigen::VectorXd last_zs){
  return (out - correct).cwiseProduct(last_zs.unaryExpr(&SigmoidPrime));
}

Eigen::VectorXd CrossEntropyCost(Eigen::VectorXd out, Eigen::VectorXd correct){
  Eigen::VectorXd output =  -correct.array() * out.array().log() +
          (Eigen::VectorXd::Ones(correct.size()) - correct).array() * (Eigen::VectorXd::Ones(out.size()) - out).array().log();
  return output;
}

Eigen::VectorXd CrossEntropyPrime(Eigen::VectorXd out, Eigen::VectorXd correct, Eigen::VectorXd last_zs){
  return out-correct;
}

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
  double eta = 0.1;
  double l = 10.;

  dataReader *train = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_training.dat", 4, 3);
  dataReader *validate = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_validation.dat", 4, 3);
  dataReader *test = new dataReader("/Users/Aman/code/NeuralNetworks/data/iris_test.dat", 4, 3);

  FFNetwork *net = new FFNetwork(topo, eta, l);
  int corr = 0;
  for (int i = 0; i < test->data->count; i++){
    Eigen::VectorXd out = net->feedForward(test->data->inputs[i], Sigmoid);
    out = out.unaryExpr(&truncate);
    if (out == test->data->outputs[i]){
      corr++;
    }
  }

  std::cout << "Before Training: " << corr << "/" << validate->data->count << " correct" << std::endl;

  int len = train->data->count;

  double goal = 0.0001;
  long max_epochs = 1e9;

  start = std::clock();
  net->Train(train->data, validate->data, goal, max_epochs, Sigmoid, SigmoidPrime, QuadCost, QuadCostPrime);
  duration = ( std::clock() - start ) / ((double) CLOCKS_PER_SEC * omp_get_max_threads());

  std::cout << "Training took " << duration << " sec" << std::endl << std::endl;


  corr = 0;
  for (int i = 0; i < test->data->count; i++){
    Eigen::VectorXd out = net->feedForward(test->data->inputs[i], Sigmoid);
    out = out.unaryExpr(&truncate);
    if (out == test->data->outputs[i]){
      corr++;
    }
  }

  std::cout << "After Training: " << corr << "/" << test->data->count << " correct" << std::endl << std::endl;

  std::cout << "Raw network output:" << std::endl;
  std::cout << net->feedForward(test->data->inputs[0], Sigmoid) << std::endl << std::endl;
  std::cout << "Truncated network output (>=0.5 = 1, <0.5 = 0):" << std::endl;
  std::cout << net->feedForward(test->data->inputs[0], Sigmoid).unaryExpr(&truncate) << std::endl << std::endl;
  std::cout << "Corresponding correct output:" << std::endl;
  std::cout << test->data->outputs[0] << std::endl;



  return 0;
}