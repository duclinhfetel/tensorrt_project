import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# create engine
conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP16)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="my_model/save_model_input_float_no_fix_shape/saved_model",
    conversion_params=conversion_params)
converter.convert()
converter.save("saved_model_trt_fp16")

# FP32
conversion_params = trt.TrtConversionParams()
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="saved_model",
    conversion_params=conversion_params)
converter.convert()
converter.save("saved_model_trt_fp32")

# evaluate
#!python3
import tensorflow as tf
import time
m = tf.saved_model.load("saved_model_trt_fp32")
ff = m.signatures['serving_default']
x = tf.ones(shape=(8,300,300,3))
y = ff(x)# It should print the following indicating that TensorRT infer libraries are loaded
# Linked TensorRT version: 7.1.3
# Successfully opened dynamic library libnvinfer.so.7
# Loaded TensorRT version: 7.1.3
# Successfully opened dynamic library libnvinfer_plugin.so.7import time
N = 1000
t1 = time.time()
for i in range(N):
  out = ff(x)
tt = time.time() - t1
print("exec time:", tt)
print(8*N/tt, "fps")

# Export saved model to frozen graph
#!python3
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2# FP16
m = tf.saved_model.load("saved_model_trt_fp16")
ff = m.signatures['serving_default']
ff = convert_variables_to_constants_v2(ff)
graph_def = ff.graph.as_graph_def(True)
tf.io.write_graph(graph_def, '.', 'model_trt_fp16.pb', as_text=False)# FP32
m = tf.saved_model.load("saved_model_trt_fp32")
ff = m.signatures['serving_default']
ff = convert_variables_to_constants_v2(ff)
graph_def = ff.graph.as_graph_def(True)
tf.io.write_graph(graph_def, '.', 'model_trt_fp32.pb', as_text=False)