import tensorflow as tf
import numpy as np
import cv2
# import tensorflow.keras.datasets as datasets
print("TensorFlow version:", tf.__version__)


def build_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="digits")
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv_1")(inputs)
    x = tf.keras.layers.MaxPooling2D(2, name="maxpooling_1")(x)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", name="conv_2")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="maxpooling_2")(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv_3")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu', name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, x, name="mnist_model")
    return model


model = build_model()
model.summary()

mnist = tf.keras.datasets.mnist
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# cv2.imwrite("test.jpg", x_test[0])

print(x_train.shape)
print(x_test.shape)

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)

model.compile(loss="categorical_crossentropy",
              optimizer=adam, metrics=["accuracy"])

batch_size = 32
epochs = 10

history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_split=0.1)
print("save model ...")
model.save("model.h5")
print("Done")

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
