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

find_package(CUDA 7.0 REQUIRED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;
            -gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;
            -gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
link_directories($ENV{CUDNN_PATH})

include_directories(kernels utility)

link_directories()

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
#        src/cuFFNetwork.cu
#        include/cuFFNetwork.h
)

cuda_add_executable(NeuralNetworks ${SOURCE_FILES})
cuda_add_cublas_to_target(NeuralNetworks)

target_link_libraries(NeuralNetworks gflags)
target_link_libraries(NeuralNetworks cudnn)