# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from shutil import copyfile
import random

dataset_dir = "C:\\Users\\Jaydev\\Documents\\GitHub\\prac_code\\tensorflow_in_practice\\course_2_CNN\\Week3\\dataset\\horse_or_human\\"
TRAINING_DIR = dataset_dir + "training\\"
TESTING_DIR = dataset_dir + "validation\\"
HORSE_SOURCE_DIR = dataset_dir + "horses\\"
HUMAN_SOURCE_DIR = dataset_dir + "humans\\"

TRAINING_HORSES_DIR = dataset_dir + "training\\horses\\"
TRAINING_HUMANS_DIR = dataset_dir + "training\\humans\\"
TESTING_HORSES_DIR = dataset_dir + "validation\\horses\\"
TESTING_HUMANS_DIR = dataset_dir + "validation\\humans\\"


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


split_size = .9
split_data(HORSE_SOURCE_DIR, TRAINING_HORSES_DIR, TESTING_HORSES_DIR, split_size)
split_data(HUMAN_SOURCE_DIR, TRAINING_HUMANS_DIR, TESTING_HUMANS_DIR, split_size)

# weights are here
# https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = "C:\\Users\\Jaydev\\Documents\\GitHub\\prac_code\\tensorflow_in_practice\\course_2_CNN\\Week3\\weights\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer("mixed10")
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(epoch)
        if (logs.get('accuracy') > 0.999):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(units=1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
validation_generator = validation_datagen.flow_from_directory(TRAINING_DIR,
                                                              target_size=(150, 150),
                                                              batch_size=32,
                                                              class_mode='binary')

history = model.fit_generator(train_generator, validation_data=(validation_generator), epochs=100, steps_per_epoch=100, verbose=2,
                    validation_steps=50, callbacks=[callbacks])


# Visualise results
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()

# Predict Results
from keras.preprocessing import image
import numpy as np
img = image.load_img("image_path", target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print("image_path" + " is a predator")
else:
    print("image_path" " is a alien")