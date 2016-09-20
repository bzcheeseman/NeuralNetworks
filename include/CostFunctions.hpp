//
// Created by Aman LaChapelle on 9/19/16.
//

#ifndef NEURALNETWORKS_COSTFUNCTIONS_HPP
#define NEURALNETWORKS_COSTFUNCTIONS_HPP

#include <Eigen/Dense>
#include "Activations.hpp"

Eigen::VectorXd QuadCost(Eigen::VectorXd out, Eigen::VectorXd correct){
  return 0.5 * (out - correct).array().pow(2);
}

Eigen::VectorXd QuadCostPrime(Eigen::VectorXd out, Eigen::VectorXd correct, Eigen::VectorXd last_zs){
  return (out - correct).cwiseProduct(last_zs.unaryExpr(&SigmoidPrime));
}

Eigen::VectorXd CrossEntropyCost(Eigen::VectorXd out, Eigen::VectorXd correct){
  Eigen::VectorXd output = -((correct.array() * out.array().log()) +
                            (Eigen::VectorXd::Ones(correct.size()) - correct).array()
                            * (Eigen::VectorXd::Ones(out.size()) - out).array().log());
  output *= 1./out.size();
  return output;
}

Eigen::VectorXd CrossEntropyPrime(Eigen::VectorXd out, Eigen::VectorXd correct, Eigen::VectorXd last_zs){
  return out-correct;
}

#endif //NEURALNETWORKS_COSTFUNCTIONS_HPP
