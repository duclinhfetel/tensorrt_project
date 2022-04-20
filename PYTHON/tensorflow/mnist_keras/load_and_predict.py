from tensorflow import keras
import numpy as np
import cv2

model = keras.models.load_model('model.h5')

model.summary()

image = cv2.imread("8.pgm")
print("Input Image From File: ", image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image/255.0
image = np.expand_dims(image, -1)
print("Input after Preprocess: ", image.shape)
output = model.predict(image[None, :, :, :])
print(output)
print("Class ID: ", np.argmax(output), ", Prob: ", np.amax(output))
