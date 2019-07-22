import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

dataset_dir = "C:\\Users\\Jaydev\\Documents\\Datasets\\tf_in_practice_datasets\\alien_vs_predator\\"
TRAINING_DIR = dataset_dir + "training\\"
TESTING_DIR = dataset_dir + "testing\\"

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    target_size=(150, 150),
                                                    batch_size=1,
                                                    class_mode='binary')

testing_datagen = ImageDataGenerator(rescale=1. / 255)
testing_generator = testing_datagen.flow_from_directory(TESTING_DIR,
                                                        target_size=(150, 150),
                                                        batch_size=100,
                                                        class_mode='binary')

history = model.fit_generator(train_generator,
                    epochs=15,
                    verbose=1,
                    steps_per_epoch=150,
                    validation_data=testing_generator)


import numpy as np
print(np.mean(history.history['acc']))
print(np.max(history.history['acc']))
# 0.71199995
# 0.8066667

print(np.mean(history.history['val_acc']))
print(np.max(history.history['val_acc']))
# 0.6563333
# 0.755

from keras.preprocessing import image
import numpy as np
image_path = "C:\\Users\\Jaydev\\Documents\\Datasets\\tf_in_practice_datasets\\alien_vs_predator\\training\\predator\\10.jpg"
img = image.load_img(image_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print(image_path + " is a predator")
else:
    print(image_path + " is a alien")

import matplotlib.pyplot as plt
def plot_graphs(history, string):
    plt.plot(history.history[string],'r')
    plt.plot(history.history['val_' + string], 'b')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "acc")
plot_graphs(history, "loss")