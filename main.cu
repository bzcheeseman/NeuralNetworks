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

#include "include/cuFFNetwork.h"

int main(int argc, char *argv[]){

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int num_inputs = 4, num_hidden = 3, num_outputs = 3;

  cuLayer hiddens(num_inputs, num_hidden);
  cuLayer outputs(num_hidden, num_outputs);

  std::cout << hiddens << std::endl;


  return 0;
}