from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import numpy as np
import tensorflow as tf
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
oov_tok = "<OOV>"
max_length = 150
padding_type = 'post'
trunc_type = 'post'

# Get the data
dataset_df = pd.read_csv(dataset_csv)
dataset_df.describe()


# Clean data from stopwords
def remove_stopwords(sentence):
    for token in stopwords:
        sentence = sentence.replace(token, " ")
        sentence = sentence.replace("  ", " ")
    return sentence


dataset_df['review'] = dataset_df.apply(remove_stopwords)

# Get tokenizers
sentence_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
sentence_tokenizer.fit_on_texts(dataset_df['review'])
word_index = sentence_tokenizer.word_index

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(dataset_df['sentiment'])
word_index = label_tokenizer.word_index

# Get datasets
from sklearn.model_selection import train_test_split

train_dataset, validation_dataset, train_label, validation_label = train_test_split(dataset_df['review'],
                                                                                    dataset_df['sentiment'],
                                                                                    train_size=0.8, random_state=0)

train_dataset, predict_dataset, train_label, predict_label = train_test_split(train_dataset, train_label,
                                                                              train_size=0.9, random_state=0)

# Apply tokenizers
train_sequences = sentence_tokenizer.texts_to_sequences(train_dataset)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

validation_sequences = sentence_tokenizer.texts_to_sequences(validation_dataset)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

predict_sequences = sentence_tokenizer.texts_to_sequences(predict_dataset)
predict_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_label))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_label))
pred_label_seq = np.array(label_tokenizer.texts_to_sequences(predict_label))

# Get Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Fit Model
num_epochs = 30
history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq), verbose=2)


# see performance
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# predict a few
