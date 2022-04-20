/**
 * @file trt_engine.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-04-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "trt_engine.h"

void Logger::log(Severity severity, const char *msg) noexcept
{
  // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
  // For the sake of this tutorial, will just log to the console.

  // Only log Warnings or more important.
  if (severity < Severity::kWARNING)
  {
    std::cout << msg << std::endl;
  }
}

bool Engine::doesFileExist(const std::string &path)
{
  std::ifstream f(path.c_str());
  return f.good();
}

Engine::Engine(const Options &options) : options_(options) {}
Engine::~Engine()
{
  if (cuda_stream_)
  {
    cudaStreamDestroy(cuda_stream_);
  }
}

bool Engine::build(std::string onnx_path)
{
  engine_name_ = serializeEngineOptions(options_);
  std::cout << "Searching for engine file with name: " << engine_name_ << std::endl;

  if (doesFileExist(engine_name_))
  {
    std::cout << "Engine found, not regenerating..." << std::endl;
    return true;
  }

  // Was not able to find the engine file, generate...
  std::cout << "Engine not found, generating..." << std::endl;

  // Create our engine builder.
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  if (!builder)
  {
    return false;
  }

  // Set the max supported batch size
  builder->setMaxBatchSize(options_.max_batch_size);

  // Define an explicit batch size and then create the network.
  // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
  auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  if (!network)
  {
    return false;
  }

  // Create a parser for reading the onnx file.
  auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  if (!parser)
  {
    return false;
  }

  // We are going to first read the onnx file into memory, then pass that buffer to the parser.
  // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
  std::ifstream file(onnx_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size))
  {
    throw std::runtime_error("Unable to read engine file");
  }

  auto parsed = parser->parse(buffer.data(), buffer.size());
  if (!parsed)
  {
    return false;
  }

  // Save the input height, width, and channels.
  // Require this info for inference.
  const auto input = network->getInput(0);
  const auto output = network->getOutput(0);
  const auto inputName = input->getName();
  const auto inputDims = input->getDimensions();
  int32_t inputC = inputDims.d[1];
  int32_t inputH = inputDims.d[2];
  int32_t inputW = inputDims.d[3];

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config)
  {
    return false;
  }

  // Specify the optimization profiles and the
  IOptimizationProfile *defaultProfile = builder->createOptimizationProfile();
  defaultProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
  defaultProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(1, inputC, inputH, inputW));
  defaultProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(options_.max_batch_size, inputC, inputH, inputW));
  config->addOptimizationProfile(defaultProfile);

  // Specify all the optimization profiles.
  for (const auto &optBatchSize : options_.opt_batch_size)
  {
    if (optBatchSize == 1)
    {
      continue;
    }

    if (optBatchSize > options_.max_batch_size)
    {
      throw std::runtime_error("optBatchSize cannot be greater than maxBatchSize!");
    }

    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
    profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(optBatchSize, inputC, inputH, inputW));
    profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(options_.max_batch_size, inputC, inputH, inputW));
    config->addOptimizationProfile(profile);
  }
  config->setMaxWorkspaceSize(options_.max_workspace_size);

  if (options_.FP16)
  {
    config->setFlag(BuilderFlag::kFP16);
  }
  if (options_.INT8)
  {
    config->setFlag(BuilderFlag::kINT8);
  }

  // CUDA stream used for profiling by the builder.
  auto profileStream = samplesCommon::makeCudaStream();
  if (!profileStream)
  {
    return false;
  }

  config->setProfileStream(*profileStream);

  // Build the engine
  std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan)
  {
    return false;
  }

  // Write the engine to disk
  std::ofstream outfile(engine_name_, std::ofstream::binary);
  outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

  std::cout << "Success, saved engine to " << engine_name_ << std::endl;
  return true;
}

bool Engine::loadNetwork()
{
  // Read the serialized model from disk
  std::ifstream file(engine_name_, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size))
  {
    throw std::runtime_error("Unable to read engine file");
  }

  std::unique_ptr<IRuntime> runtime{createInferRuntime(logger_)};
  if (!runtime)
  {
    return false;
  }

  // Set the device index
  auto ret = cudaSetDevice(options_.device_index);
  if (ret != 0)
  {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    auto errMsg = "Unable to set GPU device index to: " + std::to_string(options_.device_index) +
                  ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
    throw std::runtime_error(errMsg);
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!engine_)
  {
    return false;
  }

  contex_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!contex_)
  {
    return false;
  }

  auto cudaRet = cudaStreamCreate(&cuda_stream_);
  if (cudaRet != 0)
  {
    throw std::runtime_error("Unable to create cuda stream");
  }

  std::cout << "Load network Done\n";

  return true;
}

bool Engine::runInference(const std::vector<cv::Mat> &batch_image,
                          std::vector<std::vector<float>> &features)
{
  auto dims = engine_->getBindingDimensions(0);
  auto outputL = engine_->getBindingDimensions(1).d[1];
  std::cout << "dims: " << dims.d[0]
            << ", C: " << dims.d[1]
            << ", H: " << dims.d[2]
            << ", W: " << dims.d[3] << "\n";
  std::cout << "outputL : " << outputL << "\n";
  Dims4 inputDims = {static_cast<int32_t>(batch_image.size()), dims.d[1], dims.d[2], dims.d[3]};

  contex_->setBindingDimensions(0, inputDims);
  if (!contex_->allInputDimensionsSpecified())
  {
    throw std::runtime_error("Error, not all input dimensions specified.");
  }

  auto batchSize = static_cast<int32_t>(batch_image.size());

  // Only reallocate buffers if the batch size has changed
  if (previous_batch_size_ != batch_image.size())
  {

    input_buff_.hostBuffer.resize(inputDims);
    input_buff_.deviceBuffer.resize(inputDims);

    Dims2 outputDims{batchSize, outputL};
    output_buff_.hostBuffer.resize(outputDims);
    output_buff_.deviceBuffer.resize(outputDims);

    previous_batch_size_ = batchSize;
  }

  auto *hostDataBuffer = static_cast<float *>(input_buff_.hostBuffer.data());
  for (size_t batch = 0; batch < batch_image.size(); ++batch)
  {
    auto image = batch_image[batch];

    // Preprocess code
    image.convertTo(image, CV_32FC3, 1.f / 255.f);
    cv::subtract(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, cv::noArray(), -1);
    cv::divide(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, 1, -1);

    std::cout << "Input Shape: " << image.rows 
    << ", " << image.cols << ", " << image.channels() << "\n";
    // NHWC to NCHW conversion 
    // NHWC: For each pixel, its 3 colors are stored together in RGB order.
    // For a 3 channel image, say RGB, pixels of the R channel are stored first, then the G channel and finally the B channel.
    // https://user-images.githubusercontent.com/20233731/85104458-3928a100-b23b-11ea-9e7e-95da726fef92.png
    // int offset = dims.d[1] * dims.d[2] * dims.d[3] * batch;
    // int r = 0, g = 0, b = 0;
    // for (int i = 0; i < dims.d[1] * dims.d[2] * dims.d[3]; ++i)
    // {
    //   if (i % 3 == 0)
    //   {
    //     hostDataBuffer[offset + r++] = *(reinterpret_cast<float *>(image.data) + i);
    //   }
    //   else if (i % 3 == 1)
    //   {
    //     hostDataBuffer[offset + g++ + dims.d[2] * dims.d[3]] = *(reinterpret_cast<float *>(image.data) + i);
    //   }
    //   else
    //   {
    //     hostDataBuffer[offset + b++ + dims.d[2] * dims.d[3] * 2] = *(reinterpret_cast<float *>(image.data) + i);
    //   }
    // }
    // Host memory for input buffer
    memcpy(hostDataBuffer, image.data, image.elemSize() * image.total());
  }
  // std::cout << "finsh process convert hwc -> chw\n";
  // Copy from CPU to GPU
  auto ret = cudaMemcpyAsync(input_buff_.deviceBuffer.data(),
                             input_buff_.hostBuffer.data(),
                             input_buff_.hostBuffer.nbBytes(),
                             cudaMemcpyHostToDevice, cuda_stream_);
  if (ret != 0)
  {
    return false;
  }
  // std::cout << "copy image from cpu to mem cuda done\n";
  std::vector<void *> predicitonBindings = {input_buff_.deviceBuffer.data(),
                                            output_buff_.deviceBuffer.data()};
  // Run inference.
  
  bool status = contex_->enqueueV2(predicitonBindings.data(), cuda_stream_, nullptr);
  
  if (!status)
  {
    return false;
  }

  // Copy the results back to CPU memory
  ret = cudaMemcpyAsync(output_buff_.hostBuffer.data(),
                        output_buff_.deviceBuffer.data(),
                        output_buff_.deviceBuffer.nbBytes(),
                        cudaMemcpyDeviceToHost, cuda_stream_);
  if (ret != 0)
  {
    std::cout << "Unable to copy buffer from GPU back to CPU" << std::endl;
    return false;
  }

  ret = cudaStreamSynchronize(cuda_stream_);
  if (ret != 0)
  {
    std::cout << "Unable to synchronize cuda stream" << std::endl;
    return false;
  }

  // Copy to output
  for (int batch = 0; batch < batchSize; ++batch)
  {
    std::vector<float> featureVector;
    featureVector.resize(outputL);

    memcpy(featureVector.data(),
           reinterpret_cast<const char *>(output_buff_.hostBuffer.data()) + batch * outputL * sizeof(float),
           outputL * sizeof(float));
    features.emplace_back(std::move(featureVector));
  }
  return true;
}

std::string Engine::serializeEngineOptions(const Options &options)
{
  std::string engine_name = "trt.engine";
  std::vector<std::string> gpu_uuids;
  getGPUUUIDs(gpu_uuids);
  if (static_cast<size_t>(options.device_index) >= gpu_uuids.size())
  {
    throw std::runtime_error("Error, provided device index is out of range");
  }

  engine_name += "." + gpu_uuids[options.device_index];

  // Serialize the specified options into the filename
  if (options.FP16)
  {
    engine_name += "FP16";
  }
  else
  {
    engine_name += "FP32";
  }

  engine_name += "." + std::to_string(options.max_batch_size) + ".";
  for (size_t i = 0; i < options.opt_batch_size.size(); ++i)
  {
    engine_name += std::to_string(options.opt_batch_size[i]);
    if (i != options.opt_batch_size.size() - 1)
    {
      engine_name += "_";
    }
  }

  engine_name += "." + std::to_string(options.max_workspace_size);

  return engine_name;
}

void Engine::getGPUUUIDs(std::vector<std::string> &gpu_uuids)
{
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);
  for (int device = 0; device < num_gpus; device++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    char uuid[33];
    for (int b = 0; b < 16; b++)
    {
      sprintf(&uuid[b * 2], "%02x", (unsigned char)prop.uuid.bytes[b]);
    }
    gpu_uuids.push_back(std::string(uuid));
  }
}
// end of file