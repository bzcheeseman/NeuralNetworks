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

#include "include/cuFFNetwork.hpp"

int main(int argc, char *argv[]){

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int num_inputs = 4, num_hidden = 3, num_outputs = 3;

  cuFFLayer hiddens(num_inputs, num_hidden, 0);
  cuFFLayer outputs(num_hidden, num_outputs, 0);

  float fakein[4] = {1.2f, 1.1f, 1.3f, 0.2f};
  float fakeout[3] = {1.0f, 0.0f, 1.0f};

  std::cout << hiddens << std::endl;

  cuFFNetwork net (0, 1, hiddens, outputs);
  net.feedForward(fakein);
//  std::cout << net.feedForward(fakein) << std::endl;

  std::cout << hiddens << std::endl;

  net.backPropagate(fakeout);




  return 0;
}