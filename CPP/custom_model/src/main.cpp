#include "trt_engine.h"

int main(int argc, char const *argv[])
{
  /* code */
  // TrtEngine trt;
  // trt.loadOnnxModel("../data/model.onnx");
  Options options;
  options.opt_batch_size = {2, 4, 8};

  Engine engine(options);
  // TODO: Specify your model here.
  // Must specify a dynamic batch size when exporting the model from onnx.
  const std::string onnxModelpath = "../data/model_nhwc.onnx";

  bool succ = engine.build(onnxModelpath);
  if (!succ)
  {
    throw std::runtime_error("Unable to build TRT engine.");
  }

  succ = engine.loadNetwork();
  if (!succ)
  {
    throw std::runtime_error("Unable to load TRT engine.");
  }

  const size_t batchSize = 1;
  std::vector<cv::Mat> images;

  const std::string inputImage = "../data/8.pgm";
  auto img = cv::imread(inputImage);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  for (size_t i = 0; i < batchSize; ++i)
  {
    images.push_back(img);
  }

  // Discard the first inference time as it takes longer
  std::vector<std::vector<float>> featureVectors;
  succ = engine.runInference(images, featureVectors);
  if (!succ)
  {
    throw std::runtime_error("Unable to run inference.");
  }

  // for (auto data : featureVectors)
  // {
  double max_value = 0;
  double out_val = 0;
  int argmax = 0;
  for (int j = 0; j < batchSize; j++)
  {
    for (int i = 0; i < featureVectors[j].size(); i++)
    {
      if (featureVectors[0][i] > max_value)
      {
        max_value = featureVectors[0][i];
        argmax = i;
        out_val = featureVectors[0][i];
      }
    }
  }

  std::cout << "Class: " << argmax << ", Prob: " << out_val << "\n";
  // }

  return 0;
}
