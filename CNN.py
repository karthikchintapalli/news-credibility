import numpy as np
import re

import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2

max_seq_len = 200
emb_dim = 100
val_split = 0.2

claims = []
labels = []

with open('./data', 'r') as f:
    for line in f:
        claim = line.strip().split('\t')[0].strip('.')
        credibility = line.strip().split('\t')[1]
        claims.append(claim.lower())

        if credibility == "true" or credibility == "mostly true":
            labels.append(1)

        else:
            labels.append(0)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(claims)
sequences = tokenizer.texts_to_sequences(claims)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=max_seq_len)
labels = np.asarray(labels)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(val_split * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.random.random((len(word_index) + 1, emb_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=max_seq_len,
                            trainable=False)

convs = []
filter_sizes = [3, 4, 5]

sequence_input = Input(shape=(max_seq_len,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for size in filter_sizes:
    l_conv = Conv1D(128, size, activation='relu', kernel_regularizer=l2(0.01))(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)
    
l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_cov1 = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.01))(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_drop1 = Dropout(0.2)(l_pool1)
l_cov2 = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.01))(l_drop1)
l_pool2 = MaxPooling1D(3)(l_cov2)
l_drop2 = Dropout(0.2)(l_pool2)
l_flat = Flatten()(l_drop2)
l_dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(l_flat)
l_drop3 = Dropout(0.3)(l_dense)
preds = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(l_drop3)

model = Model(sequence_input, preds)

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val), 
            epochs=10, batch_size=10)
