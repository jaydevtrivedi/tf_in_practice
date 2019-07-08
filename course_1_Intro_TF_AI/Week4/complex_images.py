import tensorflow as tf
import os
import zipfile

DESIRED_ACCURACY = 0.999

# zipfile_path = "happy-or-sad.zip"
# zip_ref = zipfile.ZipFile(zipfile_path, 'r')
# zip_ref.extractall("h-or-s")
# zip_ref.close()


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > DESIRED_ACCURACY):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('C:\\Users\\Jaydev\\Documents\\GitHub\\prac_code\\tensorflow_in_practice\\course_1_Intro_TF_AI\\Week 4\\h-or-s',
                                              target_size=(150, 150),
                                              batch_size=32,
                                              class_mode='binary')

history = model.fit_generator(train_generator,
                              steps_per_epoch=2,
                              epochs=15,
                              verbose=1,
                              callbacks=[callbacks])
