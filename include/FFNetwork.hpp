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

/**
 * @file include/FFNetwork.hpp
 * @brief Holds a functioning FeedForward fully connected neural network.
 *
 * Holds method declarations for a functioning feed forward neural network.
 */

/**
 * @struct FFLayer include/FFNetwork.hpp
 * @brief Holds the information for one layer of a neural network.
 *
 * Encapsulates each layer of the FFNetwork class.  Makes it easier to implement dropout or convolution if one
 * is so inclined.
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

/**
 * @class FFNetwork include/FFNetwork.hpp src/FFNetwork.cpp
 * @brief Holds a basic Feed Forward fully connected neural network.
 *
 * Contains a basic fully connected Feed Forward network.  Holds training and backpropagation as well as the raw numbers
 * needed to make it work.
 */
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
   * Hides the feedForward function in favor of an overloaded operator.  Also truncates the network output which
   * feedForward(Eigen::VectorXd) does not do.
   *
   * @param input Input vector to the network
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
   * Standard Stochastic Gradient Descent - quite basic but it gets the job done.
   *
   * @param input Input to feed through the network
   * @param correct The correct output that corresponds to the input we fed in.
   * @return The gradient of the surface at the beginning of that backprop step.
   */
  double SGD(Eigen::VectorXd input, Eigen::VectorXd correct);

  /**
   * An implementation of momentum-optimized stochastic gradient descent.  There was probably a paper on this
   * but I can't find it - will happily cite it but otherwise just got the general idea from Michael Nielsen's book
   * and the equations for implementation from everywhere.
   *
   * @param input Input to feed through the network
   * @param correct The correct output that corresponds to the input we fed in.
   * @return The gradient of the surface at the beginning of that backprop step.
   */
  double MomentumSGD(Eigen::VectorXd input, Eigen::VectorXd correct);

  /**
   * Implementation of the AdaDelta backpropagation algorithm from
   *
   * ADADELTA: An Adaptive Learning Rate Method
   * Zeiler, Matthew D.
   * eprint arXiv:1212.5701
   * 12/2012
   *
   * Generally takes the fewest epochs to converge, though each epoch takes longer on average.
   *
   * @param input Input to feed through the network
   * @param correct The correct output that corresponds to the input we fed in.
   * @return The gradient of the surface at the beginning of that backprop step.
   */
  double Adadelta(Eigen::VectorXd input, Eigen::VectorXd correct);

  /**
   * Trains the network on the training and validation data provided.  The validation data is passed
   * to Evaluate(int,dataSet<double>*), the training set is used for the actual training.
   *
   * goal is the cost goal (compared against result from Evaluate(int,dataSet<double>*)
   * max_epochs is the final stopping criterion, generally set to a high value, we rely on the cost or gradient goals more.
   * min_gradient is the smallest gradient we want to have before terminating.  Basically, when the gradient (the slope)
   * is smaller than this number, we consider the ball to be at the bottom of the hill.
   *
   * @param training dataSet used for training
   * @param validation dataSet used for validation
   * @param goal Cost goal - compared against result from Evaluate(int,dataSet<double>*)
   * @param max_epochs Final stopping criterion - cuts off the training when the number of epochs hits this
   * @param min_gradient When the gradient is smaller than this number, we consider the ball to be at the bottom of the hill
   */
  void Train(dataSet<double> *training, dataSet<double> *validation, double goal, long max_epochs, double min_gradient);

  /**
   * Evaluates the neural network on a random input from the validation set.  Takes in a random number and mods it with
   * the length of the validation set to pick a testing dataset.
   *
   * @param rand_seed Random number used to choose which dataset to test on.
   * @param validation Validation dataset - labeled examples we can use to test the network's performance.
   * @return Total cost from that feedforward iteration - applies the cost function to the network output.
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
   * Writes the network details to a file.  Ignores the activation functions, etc. because that would be hard.
   *
   * TODO: Add the network functions to the file (activations, cost, regularization, dropout, derivatives)
   *
   * @param out Outgoing file stream
   * @param net The network to write to a file
   * @return Outgoing file stream
   */
  friend std::ofstream& operator<<(std::ofstream& out, const FFNetwork &net);

  /**
   * Reads from the file outputted by operator<<(std::ofstream&,FFNetwork&).  Not guaranteed to work on anything else.
   *
   * NW - really not working
   *
   * @param in Incoming file stream
   * @param filename The name of the file that holds the network details
   * @return A network initialized from the weights in the file!
   */
  friend std::ifstream& operator>>(std::ifstream& in, FFNetwork *net);
};


#endif //NEURALNETWORK_NETWORK_HPP
