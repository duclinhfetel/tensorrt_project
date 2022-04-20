cmake_minimum_required(VERSION 3.10)
project(custom_model)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

## Use C++11
set (CMAKE_CXX_STANDARD 11)


# check flags
message("Build type: ${CMAKE_BUILD_TYPE}")

# this package libs and includes for TensorRT
option(TENSORRT_FOUND "TensorRT installed on system?" OFF)

find_package(CUDA)
find_library(NVINFER  NAMES nvinfer)
find_library(NVINFERPLUGIN NAMES nvinfer_plugin)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSER NAMES nvonnxparser)


# If it is ALL there, export libraries as a single package
if(CUDA_FOUND AND NVINFER AND NVINFERPLUGIN AND NVPARSERS AND NVONNXPARSER)
  message("TensorRT available!")
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  message("NVINFER: ${NVINFER}")
  message("NVINFERPLUGIN: ${NVINFERPLUGIN}")
  message("NVPARSERS: ${NVPARSERS}")
  message("NVONNXPARSER: ${NVONNXPARSER}")
  list(APPEND TENSORRT_LIBRARIES ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers nvonnxparser)
  message("All togheter now (libs): ${TENSORRT_LIBRARIES}")
  list(APPEND TENSORRT_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  message("All togheter now (inc): ${TENSORRT_INCLUDE_DIRS}")
  set(TENSORRT_FOUND ON)
else()
  message("TensorRT NOT Available")
  set(TENSORRT_FOUND OFF)
endif()

# complain if no backend is installed
if(NOT TENSORRT_FOUND)
  message(FATAL_ERROR
  "TensorRT must be installed
  TENSORRT_FOUND ${TENSORRT_FOUND}\n")
endif()

find_package(OpenCV REQUIRED)

include_directories(
    include
    common
    ${TENSORRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

## Build executable
add_executable(
  ${PROJECT_NAME}
  src/main.cpp
  src/trt_engine.cpp
  common/logger.cpp
)
target_link_libraries(
    ${PROJECT_NAME} 
    ${TENSORRT_LIBRARIES}
    ${OpenCV_LIBRARIES}
)