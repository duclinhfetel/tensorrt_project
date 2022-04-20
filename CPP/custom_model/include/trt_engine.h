/**
 * @file trt_engine.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-04-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef TENSORRT_TRT_ENGINE_H__
#define TENSORRT_TRT_ENGINE_H__
// stard library
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>

// opencv library
#include <opencv2/opencv.hpp>

// nvidia cuda
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

// common folder
#include "buffers.h"
// #include "logger.h"

// Options for network
struct Options
{
  // use 16bit floating point type for inference
  bool FP16 = false;
  // use 8bit interger type for inference
  bool INT8 = false;
  // batch size to optimize for
  std::vector<int32_t> opt_batch_size;
  // maximum allowable batch size
  int32_t max_batch_size = 16;
  // max allowable GPU memory to be used for model conversion, in byte
  // Application should allow the engine builder as much workspace as they can afford
  // at runtime, the SDK allocates no more than this and typically less
  size_t max_workspace_size = 4000000000;
  // GPU device index
  int device_index = 0;
};

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override;
};

class Engine
{
public:
  Engine(const Options &options);
  ~Engine();

  /**
   * @brief build network from onnx file
   *
   * @param onnx_path path of onnx file
   * @return true
   * @return false
   */
  bool build(std::string onnx_path);

  /**
   * @brief Load and prepare the network for inference
   *
   * @return true
   * @return false
   */
  bool loadNetwork();

  /**
   * @brief Run inference
   *
   * @param batch_image list cv::Mat image need predict
   * @param features output of model
   * @return true
   * @return false
   */
  bool runInference(
      const std::vector<cv::Mat> &batch_image,
      std::vector<std::vector<float>> &features);

private:
  // convert the engine options into a string
  std::string serializeEngineOptions(const Options &options);

  void getGPUUUIDs(std::vector<std::string> &gpu_uuids);
  bool doesFileExist(const std::string &path);

  std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> contex_ = nullptr;
  Options options_;
  Logger logger_;
  samplesCommon::ManagedBuffer input_buff_;
  samplesCommon::ManagedBuffer output_buff_;
  // samplesCommon::ManagedBuffer m_inputBuff;
  //   samplesCommon::ManagedBuffer m_outputBuff;
  size_t previous_batch_size_ = 0;
  std::string engine_name_;
  cudaStream_t cuda_stream_ = nullptr;
};

#endif // TENSORRT_TRT_ENGINE_H__