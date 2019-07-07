import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

dataset_dir = "C:\\Users\\Jaydev\\Documents\\GitHub\\prac_code\\tensorflow_in_practice\\course_2_CNN\\Week1\\alien_vs_predator\\data\\"
# CAT_SOURCE_DIR = dataset_dir + "Cat\\"
# DOG_SOURCE_DIR = dataset_dir + "Dog\\"
TRAINING_DIR = dataset_dir + "training\\"
TESTING_DIR = dataset_dir + "testing\\"
TRAINING_ALIEN_DIR = dataset_dir + "training\\alien\\"
TRAINING_PREDATOR_DIR = dataset_dir + "training\\predator\\"
TESTING_ALIEN_DIR = dataset_dir + "testing\\alien\\"
TESTING_PREDATOR_DIR = dataset_dir + "testing\\predator\\"

#
# def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
#     files = []
#     for filename in os.listdir(SOURCE):
#         file = SOURCE + filename
#         if os.path.getsize(file) > 0:
#             files.append(filename)
#         else:
#             print(filename + " is zero length, so ignoring.")
#
#     training_length = int(len(files) * SPLIT_SIZE)
#     testing_length = int(len(files) - training_length)
#     shuffled_set = random.sample(files, len(files))
#     training_set = shuffled_set[0:training_length]
#     testing_set = shuffled_set[-testing_length:]
#
#     for filename in training_set:
#         this_file = SOURCE + filename
#         destination = TRAINING + filename
#         copyfile(this_file, destination)
#
#     for filename in testing_set:
#         this_file = SOURCE + filename
#         destination = TESTING + filename
#         copyfile(this_file, destination)
#
#
# split_size = .9
# split_data(CAT_SOURCE_DIR, TRAINING_CAT_DIR, TESTING_CAT_DIR, split_size)
# split_data(DOG_SOURCE_DIR, TRAINING_DOG_DIR, TESTING_DOG_DIR, split_size)

print(len(os.listdir(TRAINING_ALIEN_DIR)))
print(len(os.listdir(TRAINING_PREDATOR_DIR)))
print(len(os.listdir(TESTING_ALIEN_DIR)))
print(len(os.listdir(TESTING_PREDATOR_DIR)))

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

# Visualise results
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
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