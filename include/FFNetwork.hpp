//
// Created by Aman LaChapelle on 9/18/16.
//

#ifndef NEURALNETWORK_NETWORK_HPP
#define NEURALNETWORK_NETWORK_HPP

#include <Eigen/Dense>
#include <vector>
#include <ctime>
#include <random>
#include <omp.h>
#include "dataReader.hpp"
#include "DropoutandRegularization.hpp"

/*
 * TODO: read from/write to file
 * TODO: speedup for larger networks - cuDNN - look at exampke
 *  TODO: Convolution layers - probably not useful until we get to bigger networks
 * TODO: Other fancier networks!
 */

struct FFLayer{
  //! Weight between previous layer and this one
  Eigen::MatrixXd w;
  //! Bias for each layer
  Eigen::VectorXd b;
  //! The raw w.x + b
  Eigen::VectorXd z;
  //! Found by taking z and applying the activation function to it
  Eigen::VectorXd a;

  /**
   * Takes a vector from a bernoulli distribution and drops out random a's which will in turn cause neurons to drop
   * out of the next layer.
   *
   * @param r The random vector of zeros and ones that is applied to a dropout layer
   */
  inline void makeDropout(Eigen::VectorXd r){
    a = a.cwiseProduct(r);
  }
};

class FFNetwork {

  //! The number of neurons in each layer
  std::vector<unsigned> topology;
  //! Whether each layer is a dropout layer or not
  std::vector<unsigned> dropout;
  //! Learning rate - used in SGD and MOMSGD only
  double eta;
  //! Regularization rate - appears in a denominator so a value ~5e3 is recommended and speeds up training
  double lamda;
  //! Momentum parameter - a scaling parameter for MOMSGD and ADADELTA, around 0.9 is usually good
  double gamma;
  //! A small number used in ADADELTA for RMS calculation in the learning rate
  double epsilon;

  //! An array of the layers that make up the network
  FFLayer *layers;

  //! The activation function
  double (*phi)(double);
  //! Derivative of the activation function
  double (*phiprime)(double);
  //! The cost function - returns a vector that corresponds to the cost per output neuron
  Eigen::VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd);
  //! The derivative of the cost function - returns a vector that again corresponds to output neurons
  Eigen::VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);
  //! Regularization function - Identity = L2, Sign = L1, Zero = None
  double (*regularization)(double);
  //! Dropout function - currenly only suppored in DropoutandRegularization.hpp with Bernoulli
  double (*dropout_fn)(double);

  //! Chooses the backpropagation algorithm.
  enum {StochGradDescent, MOMSGD, ADADELTA}backprop;

public:
  /**
   * Constructor - sets various network parameters.  Need to call setFunctions to set the activation, cost, regularization,
   * and dropout functions.
   *
   * Default initializes the backprop algorithm to ADADELTA - so be sure to call setBackpropAlgorithm if you want to change that.
   *
   * @param topology @copydoc topology
   * @param dropout @copydoc dropout
   * @param eta @copydoc eta
   * @param lamda @copydoc lamda
   * @param gamma @copydoc gamma
   * @param epsilon @copydoc epsilon
   * @return A new neural network, mostly initialized!
   */
  FFNetwork(std::vector<unsigned> topology, std::vector<unsigned> dropout, double eta, double lamda, double gamma, double epsilon);

  /**
   * Destructor - Deletes the pointer to the layers of the network - be sure to save first!
   */
  ~FFNetwork();

  void setFunctions(double (*phi)(double),
                    double (*phiprime)(double),
                    double (*regularization)(double),
                    double (*dropout_fn)(double),
                    Eigen::VectorXd (*cost)(Eigen::VectorXd, Eigen::VectorXd),
                    Eigen::VectorXd (*costprime)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd));

  /**
   * Sets backprop algorithm.  Uses strncmp to select the algorithm.
   *
   * @param algorithm Possible inputs are "SGD", "MOMSGD", or "ADADELTA"
   */
  void setBackpropAlgorithm(const char *algorithm);

  /**
   * Makes a prediction on the network.  Uses the trained network and forward propagates one input to return one output.
   *
   * @param input
   * @return Truncated network output
   */
  Eigen::VectorXi operator()(Eigen::VectorXd input);

  /**
   * Makes a prediction based on the current network model.  Feeds an input through the network and returns the output.
   *
   * @param input Input vector to the network
   * @return Output of the network
   */
  Eigen::VectorXd feedForward(Eigen::VectorXd input);

  /**
   *
   * @param input
   * @param correct
   * @return
   */
  double SGD(Eigen::VectorXd input, Eigen::VectorXd correct);

  /**
   *
   * @param input
   * @param correct
   * @return
   */
  double MomentumSGD(Eigen::VectorXd input, Eigen::VectorXd correct);

  /**
   *
   * @param input
   * @param correct
   * @return
   */
  double Adadelta(Eigen::VectorXd input, Eigen::VectorXd correct);

  /**
   *
   * @param training
   * @param validation
   * @param goal
   * @param max_epochs
   * @param min_gradient
   */
  void Train(dataSet<double> *training, dataSet<double> *validation, double goal, long max_epochs, double min_gradient);

  /**
   *
   * @param rand_seed
   * @param validation
   * @return
   */
  double Evaluate(int rand_seed, dataSet<double> *validation);

  /**
   * Prints the network weights and biases to the stdout - careful if you use this with a large network!
   *
   * @param out
   * @param net
   * @return
   */
  friend std::ostream& operator<<(std::ostream& out, FFNetwork &net);

  /**
   * Writes the network details to a file.
   *
   * NW
   *
   * @param out Outgoing file stream
   * @param net The network to write to a file
   * @return Outgoing file stream
   */
  friend std::ofstream& operator<<(std::ofstream& out, const FFNetwork &net);

  /**
   * Reads from the file outputted by operator<<(std::ofstream&,FFNetwork&).  Not guaranteed to work on anything else.
   *
   * NW
   *
   * @param in Incoming file stream
   * @param filename The name of the file that holds the network details
   * @return A network initialized from the weights in the file!
   */
//  friend std::ifstream& operator>>(std::ifstream& in, const char *filename);
};


#endif //NEURALNETWORK_NETWORK_HPP
