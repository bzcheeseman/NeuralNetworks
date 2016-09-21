//
// Created by Aman LaChapelle on 9/18/16.
//

#include "../include/FFNetwork.hpp"

using namespace Eigen;

FFNetwork::FFNetwork(std::vector<unsigned> topology, std::vector<unsigned> dropout, double eta,
                     double lamda, double gamma, double epsilon)
        : topology(topology), dropout(dropout), eta(eta), lamda(lamda), gamma(gamma), epsilon(epsilon) {

  layers = new FFLayer[topology.size()];

  layers[0].w = MatrixXd::Zero(1,1);
  layers[0].b = VectorXd::Zero(1);

  for (int l = 1; l < topology.size(); l++){
    layers[l].w = MatrixXd::Random(topology[l], topology[l-1]) / sqrt((double)topology[l-1]);
    layers[l].b = VectorXd::Random(topology[l]) / sqrt((double)topology[l]);
    layers[l].z = VectorXd::Zero(topology[l]);
    layers[l].a = VectorXd::Zero(topology[l]);
  }

  backprop = ADADELTA;
}

FFNetwork::~FFNetwork() {
  delete layers;
}

void FFNetwork::setFunctions(double (*phi)(double),
                             double (*phiprime)(double),
                             double (*regularization)(double),
                             double (*dropout_fn)(double),
                             VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd),
                             VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)) {
  this->phi = phi;
  this->phiprime = phiprime;
  this->regularization = regularization;
  this->dropout_fn = dropout_fn;
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
  layers[0].z = VectorXd::Zero(topology[0]);
  layers[0].a = input;

  int end = (int)this->topology.size();

#pragma omp target teams distribute default(shared)
  for (int l = 1; l < end; l++){
    layers[l].z = layers[l].w * (layers[l-1].a) + layers[l].b;
    layers[l].a = (layers[l].z).unaryExpr(phi);
  }

  return layers[end-1].a;
}

double FFNetwork::SGD(VectorXd input, VectorXd correct) {

  Eigen::VectorXd r;

  for (int i = 0; i < dropout.size()-1; i++){
    if (dropout[i+1] == 1 && (i != 0 || i != dropout.size()-1)){ //drops out of the next layer - chop up the output of this layer
      r = VectorXd::Ones((layers[i].a).size()).unaryExpr(dropout_fn);
      layers[i].makeDropout(r);
    }
    else{
      ;
    }
  }

  VectorXd net_result = this->feedForward(input);
  long num_layers = topology.size();

  VectorXd delta = costprime(net_result, correct, layers[num_layers-1].z);

  double out = abs(delta.norm());

  layers[num_layers-1].b -= eta * delta;
  layers[num_layers-1].w -= eta * (delta * (layers[num_layers-2].a).transpose());

  int l;
#pragma omp target teams distribute default(shared)
  for (l = num_layers-2; l > 0; l--){
    delta = ((layers[l+1].w).transpose() * delta).cwiseProduct((layers[l].z).unaryExpr(phiprime)); //recalculate delta

    layers[l].w -= eta * (delta * (layers[l-1].a).transpose());  //update w
    layers[l].b -= eta * delta; //update b

    if (dropout[l] != 1){
      layers[l].w -= (eta/lamda) * (layers[l].w).unaryExpr(regularization); //regularization
      layers[l].b -= (eta/lamda) * (layers[l].b).unaryExpr(regularization);
    }
  }

  return out;

}

double FFNetwork::MomentumSGD(Eigen::VectorXd input, Eigen::VectorXd correct) {

  Eigen::VectorXd r;

  for (int i = 0; i < dropout.size()-1; i++){
    if (dropout[i+1] == 1 && (i != 0 || i != dropout.size()-1)){
      r = VectorXd::Ones((layers[i].a).size()).unaryExpr(dropout_fn);
      layers[i].makeDropout(r);
    }
    else{
      ;
    }
  }

  VectorXd net_result = this->feedForward(input);
  long num_layers = topology.size();

  VectorXd delta = costprime(net_result, correct, layers[num_layers-1].z);

  double out = abs(delta.norm());

  layers[num_layers-1].b -= eta * delta;
  layers[num_layers-1].w -= eta * (delta * (layers[num_layers-2].a).transpose());

  static MatrixXd gradient;

  int l;
#pragma omp target teams distribute lastprivate(gradient) default(shared)
  for (l = num_layers-2; l > 0; l--){
    delta = gamma*delta;
    delta += ((layers[l+1].w).transpose() * delta/gamma).cwiseProduct((layers[l].z).unaryExpr(phiprime)); //recalculate delta

    gradient = (delta * (layers[l-1].a).transpose());

    layers[l].w -= eta * gradient;  //update w
    layers[l].b -= eta * delta; //update b

    if (dropout[l] != 1){
      layers[l].w -= (eta/lamda) * (layers[l].w).unaryExpr(regularization); //regularization
      layers[l].b -= (eta/lamda) * (layers[l].b).unaryExpr(regularization); //regularization
    }
  }

  return out;
}

double FFNetwork::Adadelta(Eigen::VectorXd input, Eigen::VectorXd correct) {

  Eigen::VectorXd r;

  for (int i = 0; i < dropout.size()-1; i++){
    if (dropout[i+1] == 1 && (i != 0 || i != dropout.size()-1)){
      r = VectorXd::Ones((layers[i].a).size()).unaryExpr(dropout_fn);
      layers[i].makeDropout(r);
    }
    else{
      ;
    }
  }

  VectorXd net_result = this->feedForward(input);
  long num_layers = this->topology.size();

  VectorXd delta = this->costprime(net_result, correct, layers[num_layers-1].z);

  double out = abs(delta.norm());

  static double msW = 1.*delta.squaredNorm();
  static double msD = 1.*delta.squaredNorm();

  static MatrixXd gradient;

  static double lr = sqrt(msD + epsilon)/sqrt(msW + epsilon);;

  layers[num_layers-1].b -= lr * delta;
  layers[num_layers-1].w -= lr * (delta * (layers[num_layers-2].a).transpose());

  int l = num_layers-1;

#pragma omp target teams distribute
  for (l = num_layers-2; l > 0; l--){

    msD = gamma * msD + (1.-gamma)*delta.squaredNorm();

    delta = ((layers[l+1].w).transpose() * delta);
    delta = delta.cwiseProduct((layers[l].z).unaryExpr(phiprime)); //recalculate delta

    gradient = (delta * (layers[l-1].a).transpose());

    msW = gamma * msW + (1.-gamma)*gradient.squaredNorm();

    lr = sqrt(msD + epsilon)/sqrt(msW + epsilon);

    layers[l].w -= lr * gradient;  //update w
    layers[l].b -= lr * delta; //update b

    if (dropout[l] != 1){
      layers[l].w -= (lr/lamda) * (layers[l].w).unaryExpr(regularization); //regularization
      layers[l].b -= (lr/lamda) * (layers[l].b).unaryExpr(regularization);
    }
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

std::ostream &operator<<(std::ostream &out, FFNetwork &net) {
  for (int i = 0; i < net.topology.size(); i++){
    out << "========= Layer " << i << " ============" << std::endl;
    out << net.layers[i].w << std::endl << std::endl;
    out << net.layers[i].b << std::endl;
    out << "==============================" << std::endl;
  }
  out << std::endl;
  return out;
}

