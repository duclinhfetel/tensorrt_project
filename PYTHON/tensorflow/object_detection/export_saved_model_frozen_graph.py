import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2# FP16


m = tf.saved_model.load("my_model/save_model_input_float_no_fix_shape/saved_model")
ff = m.signatures['serving_default']
ff = convert_variables_to_constants_v2(ff)
graph_def = ff.graph.as_graph_def(True)
tf.io.write_graph(graph_def, '.', 'ssd_320x320_float.pb', as_text=False)





# FP32
# m = tf.saved_model.load("saved_model_trt_fp32")
# ff = m.signatures['serving_default']
# ff = convert_variables_to_constants_v2(ff)
# graph_def = ff.graph.as_graph_def(True)
# tf.io.write_graph(graph_def, '.', 'model_trt_fp32.pb', as_text=False)