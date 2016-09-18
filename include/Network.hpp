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
 * TODO: Implement regularization or something
 * TODO: read from/write to file
 * TODO: Other fancier networks!
 */

class FFNetwork {

  std::vector<unsigned> topology;
  double eta;
  double lamda;

  Eigen::MatrixXd *w;
  Eigen::VectorXd *b;

  Eigen::VectorXd *zs;
  Eigen::VectorXd *as;

  Eigen::VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd);

public:
  FFNetwork(std::vector<unsigned> topology, double eta, double lamda);
  ~FFNetwork();

  void random_double(double center, double sd, double* num){
    std::random_device rand;
    std::mt19937 generator(rand());

    std::normal_distribution<> dist(center, sd);

    num[0] = dist(generator);
  }

  static double get_sign(double in){
    if (in > 0){
      return 1.;
    }
    else if (in < 0){
      return -1.;
    }
    else{
      return 0.;
    }
  }


  Eigen::VectorXd feedForward(Eigen::VectorXd input, double (*actfunc)(double));

  void backPropagate(Eigen::VectorXd input, Eigen::VectorXd correct, double (*phi)(double),
                     double (*phiprime)(double), Eigen::VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd),
                     Eigen::VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd));

  void Train(dataSet<double> *training, dataSet<double> *validation, double goal, long max_epochs, double (*phi)(double),
             double (*phiprime)(double), Eigen::VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd),
             Eigen::VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd));

  double Evaluate(int rand_seed, dataSet<double> *validation,
                  double (*phi)(double), Eigen::VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd));
};


#endif //NEURALNETWORK_NETWORK_HPP
