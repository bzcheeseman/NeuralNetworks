//
// Created by Aman LaChapelle on 9/18/16.
//

#ifndef NEURALNETWORK_NETWORK_HPP
#define NEURALNETWORK_NETWORK_HPP

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <omp.h>
#include "dataReader.hpp"

/*
 * TODO: read from/write to file
 * TODO: speedup for larger networks - cuDNN
 * TODO: Convolutional layers
 * TODO: Dropout
 * TODO: Other fancier networks!
 */

class FFNetwork {

  std::vector<unsigned> topology;
  double eta;
  double lamda;
  double gamma;
  double epsilon;

  Eigen::MatrixXd *w;
  Eigen::VectorXd *b;

  Eigen::VectorXd *zs;
  Eigen::VectorXd *as;

  double (*phi)(double);
  double (*phiprime)(double);
  Eigen::VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd);
  Eigen::VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);
  double (*regularization)(double);

  enum {StochGradDescent, MOMSGD, ADADELTA}backprop;

public:
  FFNetwork(std::vector<unsigned> topology, double eta, double lamda, double gamma, double epsilon);
  ~FFNetwork();

  void setFunctions(double (*phi)(double),
                    double (*phiprime)(double),
                    double (*regularization)(double),
                    Eigen::VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd),
                    Eigen::VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd));

  void setBackpropAlgorithm(const char *algorithm);

//  void random_double(double center, double sd, double* num){
//    std::random_device rand;
//    std::mt19937 generator(rand());
//
//    std::normal_distribution<> dist(center, sd);
//
//    num[0] = dist(generator);
//  }

  Eigen::VectorXd feedForward(Eigen::VectorXd input);

  double SGD(Eigen::VectorXd input, Eigen::VectorXd correct);

  double MomentumSGD(Eigen::VectorXd input, Eigen::VectorXd correct);

  double Adadelta(Eigen::VectorXd input, Eigen::VectorXd correct);

  void Train(dataSet<double> *training, dataSet<double> *validation, double goal, long max_epochs, double min_gradient);

  double Evaluate(int rand_seed, dataSet<double> *validation);
};


#endif //NEURALNETWORK_NETWORK_HPP
