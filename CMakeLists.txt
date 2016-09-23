cmake_minimum_required(VERSION 3.6)
project(NeuralNetworks)

find_package(Doxygen)

if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile @ONLY)
    add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
            SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
            )
endif(DOXYGEN_FOUND)

set(CMAKE_CXX_COMPILER /usr/local/bin/g++-6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp")

find_package(eigen3 REQUIRED)
include_directories("/usr/local/include/eigen3")

set(SOURCE_FILES main.cpp
        src/FFNetwork.cpp
        include/FFNetwork.hpp
        src/dataReader.cpp
        include/dataReader.hpp
        include/Activations.hpp
        include/CostFunctions.hpp
        include/DropoutandRegularization.hpp
)

add_executable(NeuralNetworks ${SOURCE_FILES})
target_link_libraries(NeuralNetworks gflags)