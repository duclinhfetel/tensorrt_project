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


with tf.io.gfile.GFile("ssd_320x320_float.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["input_tensor:0"],
                                outputs=["Identity:0",
                                         "Identity_1:0",
                                         "Identity_2:0",
                                         "Identity_3:0",
                                         "Identity_4:0",
                                         "Identity_5:0",
                                         "Identity_6:0",
                                         "Identity_7:0"],
                                print_graph=False)

# Note that we only have "one" input and "output" for the loaded frozen function
print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)


# print("Load image")
image = cv2.imread("image.jpg")
image_backup = image.copy()
print("Input Image From File: ", image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image/255.0
# image = np.expand_dims(image, -1)

mapping_output = ['detection_anchor_indices', 'detection_boxes', 'detection_classes', 'detection_multiclass_scores',
                  'detection_scores', 'num_detections', 'raw_detection_boxes', 'raw_detection_scores']
# print(mapping_output)

image = image[None, :, :, :]
# print("Input after Preprocess: ", image.shape)
# # get prediction
output = frozen_func(
    input_tensor=tf.convert_to_tensor(image, dtype=tf.float32))

final_dict = {}

for i in range(len(mapping_output)):
    final_dict[mapping_output[i]] = output[i]

# print(final_dict['num_detections'][0])
# print(final_dict['detection_boxes'][0].numpy())
# print(final_dict['detection_classes'][0].numpy().astype(np.int32))
# print(final_dict['detection_scores'][0].numpy())


def extract(boxes, classes, scores, score_thresh):

    out_boxes = []
    out_classes = []
    out_scores = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        if scores[i] >= score_thresh:
            out_boxes.append(box)
            out_classes.append(classes[i])
            out_scores.append(scores[i])

    return out_boxes, out_classes, out_scores


def draw_output(img, boxes, classes, scores, label):
    h = img.shape[0]
    w = img.shape[1]
    for i in range(len(boxes)):
        ymin, xmin, ymax, xmax = boxes[i]
        ymin, xmin, ymax, xmax = ymin*h, xmin*w, ymax*h, xmax*w
        cv2.rectangle(img, (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)), (0, 0, 255), 4)
        cv2.putText(img, label[str(classes[i])] + " " + str(round(scores[i]*100,2))+"%", (int(xmin), int(ymin)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    return img


boxes, classes, scores = extract(final_dict['detection_boxes'][0].numpy(),
                                 final_dict['detection_classes'][0].numpy().astype(
                                     np.int32),
                                 final_dict['detection_scores'][0].numpy(), 0.7)

print(boxes, classes, scores)
image_backup = draw_output(image_backup, boxes, classes, scores, {
                           "7": "CAR", "15": "Person"})
cv2.imshow("predicts", image_backup)
cv2.waitKey(0)
#closing all open windows 
cv2.destroyAllWindows() 
# print("Class ID: ", np.argmax(output), ", Prob: ", np.amax(output))
