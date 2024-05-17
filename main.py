import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Activation
from tensorflow.python.keras.optimizers import RMSprop 

filepath = tf.python.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

# Sort, alphabetically, all characters that appears at least once in the entire text
# Each character occurs only once
characters = sorted(set(text))

# Create a dictionary that finds its index by passing a character
# i.e. {'a': 1, 'b': 2, ...etc}
# "enumerate" assigns a number to each character
char_to_index = dict((c, i) for i, c in enumerate(characters))

# Create a dictionary that finds the character by passing an index
index_to_char = dict((i, c) for i, c in enumerate(characters))

# Number of characters fed into the neural network
# in order to predict the next character
SEQ_LENGTH = 40 

# Number of characters shifted as the start of next sequence
STEP_SIZE = 3

sentences = [] # "how are yo"
next_characters = [] # "u"

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    # text[a:b] is upper-bound exclusive: starts from a through b-1, with b excluded
    sentences.append(text[i: i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])

# Convert the training data from string to numerical format
# Create an array of zeros with 3 dimensions:
# all possible sentences * individual positions in these sentences * all possible characters
x = np.zeros(len(sentences), SEQ_LENGTH, len(characters), dtype=np.bool)
y = np.zeros(len(sentences), len(characters), dtype=np.bool)

# Assign weight to each character in the text
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Feed x and y arrays into the neural network to predict the next data
model = Sequential()

# LSTM layer = for Long Short Term Memory; memory of the network; input data during all iterations
# Define the layer with 128 neurons
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))

# Followed by a dense layer
model.add(Dense(len(characters)))

# 'softmax' scales the output so all values add up to one (or 100%)
# So the 'output' is always the probability of a certain character to be the next character
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimize=RMSprop(lr=0.01))

model.fit(x, y, batch_size=256, epochs=4)