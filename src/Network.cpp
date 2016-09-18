//
// Created by Aman LaChapelle on 9/18/16.
//

#include "../include/Network.hpp"

using namespace Eigen;

FFNetwork::FFNetwork(std::vector<unsigned> topology, double eta, double lamda)
        : topology(topology), eta(eta), lamda(lamda) {

  w = new MatrixXd[topology.size()];
  b = new VectorXd[topology.size()];

  zs = new VectorXd[topology.size()];
  as = new VectorXd[topology.size()];


  w[0] = MatrixXd::Zero(1,1);
  b[0] = VectorXd::Zero(1);

  for (int l = 1; l < topology.size(); l++){
//    double rand;
//    random_double(0.0, (1./sqrt((double)topology[l-1])), &rand);
    w[l] = MatrixXd::Random(topology[l], topology[l-1]);
    b[l] = VectorXd::Random(topology[l]);
    zs[l] = VectorXd::Zero(topology[l]);
    as[l] = VectorXd::Zero(topology[l]);
  }
}

FFNetwork::~FFNetwork() {
  ;
}

Eigen::VectorXd FFNetwork::feedForward(Eigen::VectorXd input, double (*actfunc)(double)) {
  zs[0] = VectorXd::Zero(topology[0]);
  as[0] = input;

  int end = (int)this->topology.size();

  for (int l = 1; l < end; l++){
    zs[l] = w[l] * as[l-1] + b[l];
    as[l] = zs[l].unaryExpr(actfunc);
  }

  return as[end-1];
}

void FFNetwork::backPropagate(VectorXd input, VectorXd correct, double (*phi)(double),
                            double (*phiprime)(double), VectorXd (*cost)(VectorXd, VectorXd),
                            VectorXd (*costprime)(VectorXd, VectorXd, VectorXd)) {

  VectorXd net_result = this->feedForward(input, phi);
  long layers = topology.size();

  VectorXd delta = costprime(net_result, correct, zs[layers-1]);
  b[layers-1] -= eta * delta;
  w[layers-1] -= eta * (delta * as[layers-2].transpose());

#pragma omp target teams distribute
  for (int l = layers-2; l > 0; l--){
    delta = (w[l+1].transpose() * delta).cwiseProduct(zs[l].unaryExpr(phiprime)); //recalculate delta

    w[l] -= eta * (delta * as[l-1].transpose());  //update w
//    w[l] -= (eta/lamda) * w[l]; //regularization
    b[l] -= eta * delta; //update b
  }

}

void FFNetwork::Train(dataSet<double> *training, dataSet<double> *validation, double goal, long max_epochs, double (*phi)(double),
                    double (*phiprime)(double), VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd),
                    VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)) {

  long epochs = 0;

  std::random_device rand;
  static thread_local std::mt19937 generator(rand());
  std::uniform_int_distribution<> dist(0, training->count-1);

  std::cout << "Initial cost on validation set: " << Evaluate(dist(generator), validation, phi, cost) << std::endl;

  bool abort = false;

  static int choose;
  static double performance;

#pragma omp parallel for
  for (long i = 0; i <= max_epochs; i++){

    static std::mt19937 gen;

#pragma omp flush(abort)
    if (!abort) {
#pragma omp atomic update
      epochs++;

#pragma omp atomic write
      choose = dist(gen);
#pragma omp atomic write
      performance = Evaluate(choose, validation, phi, cost);

      if (performance < goal) {
        abort = true;
#pragma omp flush(abort)
      }

      this->backPropagate(training->inputs[choose], training->outputs[choose], phi, phiprime, cost, costprime);
    }

  }


  std::cout << "Finished training in " << epochs << " epochs" << std::endl;
  std::cout << "Final cost on validation set: " << Evaluate(dist(generator), validation, phi, cost) << std::endl << std::endl;

}

double FFNetwork::Evaluate(int rand_seed, dataSet<double> *validation, double (*phi)(double),
                         VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd)) {

  VectorXd in = validation->inputs[rand_seed%validation->count];
  VectorXd netOut = feedForward(in, phi);
  VectorXd corr = validation->outputs[rand_seed%validation->count];

  double TotalCost = cost(netOut, corr).norm();

  return TotalCost;
}
