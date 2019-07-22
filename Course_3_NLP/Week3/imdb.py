from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import numpy as np
import tensorflow as tf

print(tf.__version__)
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset_csv = "C:\\Users\\Jaydev\\Documents\\Datasets\\tf_in_practice_datasets\\imdb_dataset\\reviews.csv"
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]

vocab_size = 1000
embedding_dim = 16
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 2500
end = 3000

# Get the data
dataset_df = pd.read_csv(dataset_csv)

# # Get datasets
from sklearn.model_selection import train_test_split

# train_dataset, extra_dataset, train_label, extra_labels = train_test_split(dataset_df['review'],
#                                                                            dataset_df['sentiment'],
#                                                                            train_size=0.1, random_state=0)

train_dataset, validation_dataset, train_label, validation_label = train_test_split(dataset_df['review'],
                                                                                    dataset_df['sentiment'],
                                                                                    train_size=0.9, random_state=0)

train_dataset, predict_dataset, train_label, predict_label = train_test_split(train_dataset, train_label,
                                                                              train_size=0.8, random_state=0)


# Clean data from stopwords
def remove_stopwords(sentence):
    for token in stopwords:
        token = " " + token + " "
        sentence = sentence.lower()
        sentence = sentence.replace(token, " ")
        sentence = sentence.replace("  ", " ")
    return sentence


train_dataset = train_dataset.apply(remove_stopwords)
validation_dataset = validation_dataset.apply(remove_stopwords)
predict_dataset = predict_dataset.apply(remove_stopwords)

# Get tokenizers
sentence_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
sentence_tokenizer.fit_on_texts(dataset_df['review'])
word_index = sentence_tokenizer.word_index
print(len(word_index))

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(dataset_df['sentiment'])

# Apply tokenizers
train_dataset = sentence_tokenizer.texts_to_sequences(train_dataset)
train_dataset_padded = pad_sequences(train_dataset, padding=padding_type, maxlen=max_length, truncating=trunc_type)

validation_dataset = sentence_tokenizer.texts_to_sequences(validation_dataset)
validation_dataset_padded = pad_sequences(validation_dataset, padding=padding_type, maxlen=max_length,
                                          truncating=trunc_type)

predict_dataset = sentence_tokenizer.texts_to_sequences(predict_dataset)
predict_dataset_padded = pad_sequences(predict_dataset, padding=padding_type, maxlen=max_length, truncating=trunc_type)

train_label = np.array(label_tokenizer.texts_to_sequences(train_label))
train_label = train_label - 1

validation_label = np.array(label_tokenizer.texts_to_sequences(validation_label))
validation_label = validation_label - 1

predict_label = np.array(label_tokenizer.texts_to_sequences(predict_label))
predict_label = predict_label - 1


def single_layer_lstm(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# imdb dataset : R2 score is 0.24971122423292957

def multi_layer_lstm(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# imdb dataset : R2 score is 0.10465249519343711

def conv1D(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# imdb dataset : R2 score is 0.316332023545289

def bidirectional_lstm(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# imdb dataset : R2 score is 0.348930695799229

def conv1d_sarc(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# imdb dataset : R2 score is 0.4266507584287702

def multilayer_gru(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# imdb dataset : R2 score is 0.316332023545289

# Fit Model
num_epochs = 30
early_stop = tf.keras.callbacks.EarlyStopping(monitor=['accuracy'], patience=3, verbose=2)
model = conv1D(vocab_size)
history = model.fit(train_dataset_padded, train_label, epochs=num_epochs,
                    validation_data=(validation_dataset_padded, validation_label), verbose=2)

# Step 6 :  R2 Score
from sklearn.metrics import r2_score

print("R2 score is {}".format(r2_score(predict_label, model.predict(predict_dataset_padded))))

# Step 7 : Predict a few

# plot graphs
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
