//
// Created by Aman LaChapelle on 9/19/16.
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

#ifndef NEURALNETWORKS_COSTFUNCTIONS_HPP
#define NEURALNETWORKS_COSTFUNCTIONS_HPP

#include <Eigen/Dense>
#include "Activations.hpp"

inline Eigen::VectorXd QuadCost(Eigen::VectorXd out, Eigen::VectorXd correct){
  return 0.5 * (out - correct).array().pow(2);
}

inline Eigen::VectorXd QuadCostPrime(Eigen::VectorXd out, Eigen::VectorXd correct, Eigen::VectorXd last_zs){
  return (out - correct).cwiseProduct(last_zs.unaryExpr(&SigmoidPrime));
}

inline Eigen::VectorXd CrossEntropyCost(Eigen::VectorXd out, Eigen::VectorXd correct){
  Eigen::VectorXd output = -((correct.array() * out.array().log()) +
                            (Eigen::VectorXd::Ones(correct.size()) - correct).array()
                            * (Eigen::VectorXd::Ones(out.size()) - out).array().log());
  output *= 1./out.size();
  return output;
}

inline Eigen::VectorXd CrossEntropyPrime(Eigen::VectorXd out, Eigen::VectorXd correct, Eigen::VectorXd last_zs){
  return out-correct;
}

#endif //NEURALNETWORKS_COSTFUNCTIONS_HPP
