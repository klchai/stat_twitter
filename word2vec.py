import pandas as pd
import numpy as np
import time
import gensim
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences

X_orig=[]
y_orig=[]

with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,tag,company = metadata.split(",")
        company=company[:-1]
        X_orig.append(tweet)
        y_orig.append(tag)

vocab_size = 2000
maxlen = 140
t = Tokenizer(vocab_size)
tik = time.time()
t.fit_on_texts(X_orig)
tok = time.time()
word_index = t.word_index
print("All vocab size:",len(word_index))
print("Fitting time: ",(tok-tik),'s')
print("Start Vectorizing the sentences...")
v_X = t.texts_to_sequences(X_orig)
print("Start padding...")
pad_X = pad_sequences(v_X, maxlen=maxlen, padding='post')
print("Finished")

# https://github.com/mmihaltz/word2vec-GoogleNews-vectors

model_file = './GoogleNews-vectors-negative300.bin'
print("Loading word2vec model...")

wv_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

embedding_matrix = np.random.uniform(size=(vocab_size+1, 300))
print("Transfer to embedding matrix...")
for w,i in word_index.items():
    try:
        word_vector = wv_model[w]
        embedding_matrix[i] = word_vector
    except:
        print("Word: [",w,"] not in wvmodel. Use random embedding.")
print("Finished")
print("Embedding matrix shape:\n",embedding_matrix.shape)