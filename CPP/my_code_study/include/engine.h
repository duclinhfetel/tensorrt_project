/**
 * @file engine.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-04-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef ENGINE_H__
#define ENGINE_H__

#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "buffers.h"
#include "NvOnnxParser.h"

#include <iostream>
#include <fstream>
#include <chrono>



// Options for the network
struct Options
{
  // Use 16 bit floating point type for inference
  bool FP16 = false;
  // Batch sizes to optimize for.
  std::vector<int32_t> optBatchSizes;
  // Maximum allowable batch size
  int32_t maxBatchSize = 4;
  // Max allowable GPU memory to be used for model conversion, in bytes.
  // Applications should allow the engine builder as much workspace as they can afford;
  // at runtime, the SDK allocates no more than this and typically less.
  size_t maxWorkspaceSize = 4000000000;
  // GPU device index
  int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override;
};

class Engine
{
public:
  Engine(const Options &options);
  ~Engine();

  // Build the network
  bool build(std::string onnxModelPath);

  // Load and prepare the network for inference
  bool loadNetwork();

  // Run inference.
  bool runInference(const std::vector<cv::Mat> &inputFaceChips, std::vector<std::vector<float>> &featureVectors);

private:
  // Converts the engine options into a string
  std::string serializeEngineOptions(const Options &options);

  void getGPUUUIDs(std::vector<std::string> &gpuUUIDs);

  bool doesFileExist(const std::string &filepath);

  std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
  const Options &m_options;
  Logger m_logger;
  samplesCommon::ManagedBuffer m_inputBuff;
  samplesCommon::ManagedBuffer m_outputBuff;
  size_t m_prevBatchSize = 0;
  std::string m_engineName;
  cudaStream_t m_cudaStream = nullptr;
};

#endif // ENGINE_H__