import tensorflow as tf
import numpy as np
import cv2


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=True)

# Note that we only have "one" input and "output" for the loaded frozen function
print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)


print("Load image")
image = cv2.imread("8.pgm")
print("Input Image From File: ", image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image/255.0
image = np.expand_dims(image, -1)

image = image[None, :, :, :]
print("Input after Preprocess: ", image.shape)
# get prediction
output = frozen_func(x=tf.convert_to_tensor(image, dtype=tf.float32))

print(output)
print("Class ID: ", np.argmax(output), ", Prob: ", np.amax(output))
