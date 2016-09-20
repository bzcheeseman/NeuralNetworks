cmake_minimum_required(VERSION 3.6)
project(NeuralNetworks)

set(CMAKE_CXX_COMPILER /usr/local/bin/g++-6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp")

find_package(eigen3 REQUIRED)
include_directories("/usr/local/include/eigen3")

set(SOURCE_FILES main.cpp src/FFNetwork.cpp include/FFNetwork.hpp src/dataReader.cpp include/dataReader.hpp include/Activations.hpp include/CostFunctions.hpp)
add_executable(NeuralNetworks ${SOURCE_FILES})
target_link_libraries(NeuralNetworks gflags)