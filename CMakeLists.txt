cmake_minimum_required(VERSION 3.6)
project(NeuralNetworks)

# Gotta un-hide the 7.3.1 developer tools - mv /Library/Developer/CommandLineTools7.3 /Library/Developer/CommandLineTools
# and then mv /Library/Developer/CommandLineTools /Library/Developer/CommandLineTools7.3 when done

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

find_package(CUDA 8.0 REQUIRED)

#debug
#-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=compute_52;-gencode;arch=compute_50,code=compute_50;

#non-debug
#-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(/usr/local/cuda/include /usr/local/cuda)
link_directories(/usr/local/cuda/lib)
link_directories(/usr/local/lib /usr/lib)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(eigen3 REQUIRED)
include_directories(/usr/local/include/eigen3)

include_directories(/usr/local/include/gflags)

set(SOURCE_FILES
#        main.cpp
#        src/FFNetwork.cpp
#        include/FFNetwork.hpp
#        src/dataReader.cpp
#        include/dataReader.hpp
#        include/Activations.hpp
#        include/CostFunctions.hpp
#        include/DropoutandRegularization.hpp
        src/cuFFNetwork.cu
        include/cuFFNetwork.hpp
        src/cudaKernels.cu
        include/cudaKernels.hpp
        main.cu
)

cuda_add_executable(NeuralNetworks ${SOURCE_FILES})
cuda_add_cublas_to_target(NeuralNetworks)

target_link_libraries(NeuralNetworks gflags cudnn curand)