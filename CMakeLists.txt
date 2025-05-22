cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project( cuda_mlp LANGUAGES CUDA CXX )

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(${CMAKE_PROJECT_NAME}
	           main.cu
               nn_utils/bce_cost.cu
			   layers/linear_layer.cu
			   nn_utils/matrix.cu
			   neural_network.cu
			   layers/relu_activation.cu
			   nn_utils/shape.cu
			   layers/sigmoid_activation.cu
			   coordinates_dataset.cu)
			   