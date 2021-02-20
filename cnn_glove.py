import pandas as pd
import numpy as np
import time
import os
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.applications import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X_orig=[]
y_orig=[]
le = LabelEncoder()

with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,tag,company = metadata.split(",")
        company=company[:-1]
        X_orig.append(tweet)
        y_orig.append(tag)

le.fit(y_orig)
y_orig = le.transform(y_orig)

print("Total %s tweets." % len(X_orig))
print("Tweet:",X_orig[0])
print("Label:",y_orig[0])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_orig)
sequences = tokenizer.texts_to_sequences(X_orig)

word_index = tokenizer.word_index
print("Total %s unique tokens." % len(word_index))

data = pad_sequences(sequences)
print("Data tensor:",data.shape)

X_train, X_test, y_train, y_test = train_test_split(data,y_orig,test_size=0.2,random_state=0)
print("X_Train:",X_train[0])

# Import embeddings: https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
embeddings_index = {}
f = open(os.path.join('./', 'glove.6B.100d.txt'),'r',encoding='utf-8')
for line in f.readlines():
    val = line.split()
    tok = val[0]
    coefs = np.asarray(val[1:], dtype='float32')
    embeddings_index[tok] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_mtx = np.zeros((len(word_index) + 1,100))
for tok, i in word_index.items():
    embedding_vec = embeddings_index.get(tok)
    if embedding_vec is not None:
        embedding_mtx[i] = embedding_vec
print(embedding_mtx)

embedding_layer = Embedding(len(word_index)+1, 100, weights=[embedding_mtx], input_length=31, trainable=False)
seq_input = Input(shape=(31,), dtype='int32')
embedded_seq = embedding_layer(seq_input)
print(embedded_seq.shape)

x0 = Conv1D(128, 1, activation='relu')(embedded_seq)
x1 = MaxPooling1D(1)(x0)
x2 = Conv1D(128, 1, activation='relu')(x1)
x3 = MaxPooling1D(1)(x2)
x4 = Conv1D(128, 1, activation='relu')(x3)
x5 = MaxPooling1D(31)(x4)
x6 = Flatten()(x5)
x_final = Dense(128, activation='relu')(x6)
preds = Dense(1, activation='sigmoid')(x_final)

model = Model(seq_input, preds)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128)