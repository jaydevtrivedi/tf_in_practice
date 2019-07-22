import json
import tensorflow as tf
import csv
import random
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

embedding_dim = 16
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 160000
test_portion = .1
vocab_size = 1000

file = "C:\\Users\\Jaydev\\Documents\\Datasets\\tf_in_practice_datasets\\stanford_emoticon\\training_cleaned.csv"

dataframe = pd.read_csv(file)
sentences = dataframe.iloc[:, 5].values
labels = dataframe.iloc[:, 0].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

from sklearn.model_selection import train_test_split

training_sequences, test_sequences, training_labels, test_labels = train_test_split(sentences, labels,
                                                                                    test_size=test_portion,
                                                                                    random_state=0)

training_sequences = tokenizer.texts_to_sequences(training_sequences)
train_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
training_labels = training_labels / 4.0

test_sequences = tokenizer.texts_to_sequences(test_sequences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_labels = test_labels / 4.0

glove_file = "C:\\Users\\Jaydev\\Documents\\Models\\Glove\\glove.6B.100d.txt"
embeddings_index = {}
with open(glove_file, encoding="utf8") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;

print(len(embeddings_matrix))
# Expected Output
# 138859

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length, weights=[embeddings_matrix],
                              trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 2
early_stop = tf.keras.callbacks.EarlyStopping(monitor=['accuracy'], patience=3)
history = model.fit(train_padded, training_labels, epochs=num_epochs, validation_data=(test_padded, test_labels),
                    verbose=2)

print("Training Complete")

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()

# Expected Output
# A chart where the validation loss does not increase sharply!
