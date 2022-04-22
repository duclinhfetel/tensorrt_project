from __future__ import print_function

import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os
import common

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 30  # 1GiB
            config.set_flag(trt.BuilderFlag.FP16)
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                    onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 320, 320, 3]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(
                onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'ssd_320x320_float.onnx'
    engine_file_path = "ssd_320x320_float.trt"
    image_path = "image.jpg"

    input_resolution_hw = (320, 320)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (320,320))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # normalize data
    # image = image/255.
    #convert (h,w) -> (h,w,1)
    # image = np.expand_dims(image, axis=-1)
    # HWC to CHW format:
    # image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.array(image, dtype=np.float32, order='C')
    print(image.shape)
    # print(image[0][0])
    
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
      inputs, outputs, bindings, stream = common.allocate_buffers(engine)
      print('Running inference on image {}...'.format(image_path))
      inputs[0].host = image
      trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(trt_outputs)
    # print("Class ID: ", np.argmax(trt_outputs), ", Prob: ", np.amax(trt_outputs))
if __name__ == '__main__':
    main()
