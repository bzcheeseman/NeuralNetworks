//
// Created by Aman LaChapelle on 9/18/16.
//

#include "../include/FFNetwork.hpp"

using namespace Eigen;

FFNetwork::FFNetwork(std::vector<unsigned> topology, double eta, double lamda, double gamma, double epsilon)
        : topology(topology), eta(eta), lamda(lamda), gamma(gamma), epsilon(epsilon) {

  w = new MatrixXd[topology.size()];
  b = new VectorXd[topology.size()];

  zs = new VectorXd[topology.size()];
  as = new VectorXd[topology.size()];


  w[0] = MatrixXd::Zero(1,1);
  b[0] = VectorXd::Zero(1);

  for (int l = 1; l < topology.size(); l++){
    w[l] = MatrixXd::Random(topology[l], topology[l-1]) / sqrt((double)topology[l-1]);
    b[l] = VectorXd::Random(topology[l]) / sqrt((double)topology[l]);
    zs[l] = VectorXd::Zero(topology[l]);
    as[l] = VectorXd::Zero(topology[l]);
  }
}

FFNetwork::~FFNetwork() {
  ;
}

void FFNetwork::setFunctions(double (*phi)(double), double (*phiprime)(double), double (*regularization)(double),
                             VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd),
                             VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)) {
  this->phi = phi;
  this->phiprime = phiprime;
  this->regularization = regularization;
  this->cost = cost;
  this->costprime = costprime;
}

void FFNetwork::setBackpropAlgorithm(const char *algorithm) {
  int n = strlen(algorithm);
  if (strncmp(algorithm, "SGD", n) == 0){
    this->backprop = StochGradDescent;
  }
  else if (strncmp(algorithm, "MOMSGD", n) == 0){
    this->backprop = MOMSGD;
  }
  else if (strncmp(algorithm, "ADADELTA", n) == 0){
    this->backprop = ADADELTA;
  }
}

Eigen::VectorXd FFNetwork::feedForward(Eigen::VectorXd input) {
  zs[0] = VectorXd::Zero(topology[0]);
  as[0] = input;

  int end = (int)this->topology.size();

#pragma omp target teams distribute default(shared)
  for (int l = 1; l < end; l++){
    zs[l] = w[l] * as[l-1] + b[l];
    as[l] = zs[l].unaryExpr(phi);
  }

  return as[end-1];
}

double FFNetwork::SGD(VectorXd input, VectorXd correct) {

  VectorXd net_result = this->feedForward(input);
  long layers = topology.size();

  VectorXd delta = costprime(net_result, correct, zs[layers-1]);

  double out = abs(delta.norm());

  b[layers-1] -= eta * delta;
  w[layers-1] -= eta * (delta * as[layers-2].transpose());

  int l;
#pragma omp target teams distribute default(shared)
  for (l = layers-2; l > 0; l--){
    delta = (w[l+1].transpose() * delta).cwiseProduct(zs[l].unaryExpr(phiprime)); //recalculate delta

    w[l] -= eta * (delta * as[l-1].transpose());  //update w
    w[l] -= (eta/lamda) * w[l].unaryExpr(regularization); //regularization
    b[l] -= eta * delta; //update b

  }

  return out;

}

double FFNetwork::MomentumSGD(Eigen::VectorXd input, Eigen::VectorXd correct) {

  VectorXd net_result = this->feedForward(input);
  long layers = topology.size();

  VectorXd delta = costprime(net_result, correct, zs[layers-1]);

  double out = abs(delta.norm());

  b[layers-1] -= eta * delta;
  w[layers-1] -= eta * (delta * as[layers-2].transpose());

  static MatrixXd gradient;

  int l;
#pragma omp target teams distribute lastprivate(gradient) default(shared)
  for (l = layers-2; l > 0; l--){
    delta = gamma*delta;
    delta += (w[l+1].transpose() * delta/gamma).cwiseProduct(zs[l].unaryExpr(phiprime)); //recalculate delta

    gradient = (delta * as[l-1].transpose());

    w[l] -= eta * gradient;  //update w
    w[l] -= (eta/lamda) * w[l].unaryExpr(regularization); //regularization
    b[l] -= eta * delta; //update b

  }

  return out;
}

double FFNetwork::Adadelta(Eigen::VectorXd input, Eigen::VectorXd correct) {

  VectorXd net_result = this->feedForward(input);
  long layers = topology.size();

  VectorXd delta = costprime(net_result, correct, zs[layers-1]);

  double out = abs(delta.norm());

  b[layers-1] -= eta * delta;
  w[layers-1] -= eta * (delta * as[layers-2].transpose());

  static double msW = 10.*delta.squaredNorm();
  static double msD = 10.*delta.squaredNorm();

  static MatrixXd gradient;

  static double lr;

  int l = layers-1;

#pragma omp target teams distribute
  for (l = layers-2; l > 0; l--){

    msD = gamma * msD + (1.-gamma)*delta.squaredNorm();

    delta = (w[l+1].transpose() * delta).cwiseProduct(zs[l].unaryExpr(phiprime)); //recalculate delta

    gradient = (delta * as[l-1].transpose());

    msW = gamma * msW + (1.-gamma)*gradient.squaredNorm();

    lr = sqrt(msD + epsilon)/sqrt(msW + epsilon);

    w[l] -= lr * gradient;  //update w
    w[l] -= lr/lamda * w[l].unaryExpr(regularization); //regularization
    b[l] -= lr * delta; //update b

  }

  return out;
}

void FFNetwork::Train(dataSet<double> *training, dataSet<double> *validation, double goal,
                      long max_epochs, double min_gradient) {

  long epochs = 0;

  std::random_device rand;
  static thread_local std::mt19937 generator(rand());
  std::uniform_int_distribution<> dist(0, training->count-1);

  std::cout << "Initial cost on validation set: " << Evaluate(dist(generator), validation) << std::endl;

  bool abort_goal_reached = false;
  bool abort_gradient_reached = false;

  static int choose;
  static double performance;


  for (long i = 0; i <= max_epochs; i++){

    static std::mt19937 gen;

    if (!abort_goal_reached && !abort_gradient_reached) {
      epochs++;
      choose = dist(gen);
      performance = Evaluate(choose, validation);

      if (performance < goal) {
        abort_goal_reached = true;
      }

      double gradient;
      if (this->backprop == StochGradDescent){
        gradient = this->SGD(training->inputs[choose], training->outputs[choose]);
      }
      else if (this->backprop == MOMSGD){
        gradient = this->MomentumSGD(training->inputs[choose], training->outputs[choose]);
      }
      else if (this->backprop == ADADELTA){
        gradient = this->Adadelta(training->inputs[choose], training->outputs[choose]);
      }

      if (gradient < min_gradient){
        abort_gradient_reached = true;
      }
    }
  }

  if (abort_goal_reached){
    std::cout << "Finished training in " << epochs << " epochs, cost goal reached" << std::endl;
  }
  else if (abort_gradient_reached){
    std::cout << "Finished training in " << epochs << " epochs, gradient < MIN_GRADIENT = " << min_gradient << std::endl;
  }

  std::cout << "Final cost on validation set: " << Evaluate(dist(generator), validation) << std::endl << std::endl;

}

double FFNetwork::Evaluate(int rand_seed, dataSet<double> *validation) {

  VectorXd in = validation->inputs[rand_seed%validation->count];
  VectorXd netOut = feedForward(in);
  VectorXd corr = validation->outputs[rand_seed%validation->count];

  double TotalCost = cost(netOut, corr).norm();

  return TotalCost;
}

