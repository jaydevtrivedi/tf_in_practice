import tensorflow as tf

dataset_dir = "C:\\Users\\Jaydev\\Documents\\GitHub\\prac_code\\tensorflow_in_practice\\course_2_CNN\\Week4\\dataset\\"
TRAINING_DIR = dataset_dir + "training\\"
TESTING_DIR = dataset_dir + "validation\\"


from keras.preprocessing.image import ImageDataGenerator

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
                                                    batch_size=50,
                                                    class_mode='categorical')

from keras.preprocessing.image import ImageDataGenerator

validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
validation_generator = train_datagen.flow_from_directory(TESTING_DIR,
                                                         target_size=(150, 150),
                                                         batch_size=50,
                                                         class_mode='categorical')

from tensorflow.keras.models import Sequential

model = Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator, validation_data=validation_generator, steps_per_epoch=100, epochs=100, validation_steps=50, verbose=2)