import pandas as pd
import numpy as np
from numpy import dstack

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from keras.utils import np_utils, plot_model
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

all_words = set()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenize(tweet,tag=None):
    tokens=[]
    ponctuation=[".",";","!",",","-","'",'"',"\n"]
    for p in ponctuation:
        tweet=tweet.replace(p," ")
    for word in tweet.split():
        if word.isdigit():
            continue
        else:
            contains_digit=any(map(str.isdigit,word))
            if contains_digit:
                continue
            elif word.startswith("http://") or word.startswith("https://"):
                continue
            elif word.lower() in stop_words:
                continue
            elif word[0]=="@":
                continue
            elif word[0]=="#":
                word=word[1:]
                start_index=0
                for i,letter in enumerate(word):
                    if letter.isupper():
                        if i!=0:
                            new_word = lemmatizer.lemmatize(word[start_index:i].lower())
                            tokens.append(new_word)
                            if tag is not None and tag != "irr":
                                all_words.add(new_word)
                        start_index=i
                            
                    elif i==len(word)-1:
                        new_word = lemmatizer.lemmatize(word[start_index:].lower())
                        tokens.append(new_word)
                        if tag is not None and tag != "irr":
                            all_words.add(new_word)
                    else:
                        continue
            else:
                new_word = lemmatizer.lemmatize(word.lower())
                tokens.append(new_word)
                if tag is not None and tag != "irr":
                    all_words.add(new_word)
    return tokens

X_svm = []
y_svm = []
X_nn = []
y_nn = []
with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,tag,_ = metadata.split(",")
        tokens = tokenize(tweet,tag)
        X_svm.append(tokens)
        if tag != "irr":
            y_svm.append("rel")
            X_nn.append(tokens)
            y_nn.append(tag)
        else:
            y_svm.append(tag)

print("All tweets are loaded.")
# SVM
tweets_svm = [" ".join(tokens) for tokens in X_svm]
tfidf = TfidfVectorizer(min_df=1, norm='l2', ngram_range=(1, 3), stop_words='english')

print("Creating Vectors of tweets...")
features = tfidf.fit_transform(tweets_svm)
print("Vectors shape:",features.shape)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(features, y_svm, test_size=0.2, random_state=0)

svc = svm.SVC(kernel='rbf', class_weight='balanced')
print("Fitting SVM Classifier...")
svc.fit(X_train_svm, y_train_svm)
y_pred = svc.predict(X_test_svm)
print(classification_report(y_test_svm, y_pred))

# Neural network
tweets_nn = np.array([" ".join(tokens) for tokens in X_nn])
X_nn = np.array([" ".join(tokens) for tokens in X_nn])

# coder les valeurs de classe en int
le = LabelEncoder()
le.fit(y_nn)
y_nn = le.transform(y_nn)

# convertir des entiers en OneHot Encode
dummy_y = np_utils.to_categorical(y_nn)

# convertir des entiers en classe
encod_res = {0:'neg', 1:'neu', 2:'pos'}

# calculer les poids entre les différentes classes 
weight = compute_class_weight(class_weight='balanced', classes=[0,1,2], y=y_nn)
print("Class weight:", weight)

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, dummy_y, test_size=0.2, random_state=0)

max_sequence_length = max([len(tokens) for tokens in X_nn])
MAX_TOKENS_NUM = len(all_words)
EMBEDDING_DIMS = 120

vectorize_layer = TextVectorization(
  max_tokens=MAX_TOKENS_NUM,
  output_mode='int',
  output_sequence_length=max_sequence_length
)
vectorize_layer.adapt(tweets_nn)

model = Sequential()
model.add(Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(layers.Embedding(MAX_TOKENS_NUM,EMBEDDING_DIMS))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print("Fitting Neural Network...")
model.fit(X_train_nn, y_train_nn, epochs=10, class_weight=dict(enumerate(weight)))

loss, accuracy = model.evaluate(X_train_nn, y_train_nn, verbose=False)
print("Training Accuracy: {:.4f} Loss: {:.4f}".format(accuracy, loss))
loss, accuracy = model.evaluate(X_test_nn, y_test_nn, verbose=False)
print("Testing Accuracy:  {:.4f} Loss: {:.4f}".format(accuracy, loss))

def testset():
    X_testset = []
    ids_list = []
    comp_list = []
    testset_orig = []
    with open("./test.txt","r") as file:
        for line in file:
            metadata,tweet = line[:14],line[14:]
            testset_orig.append(tweet)

            ids,_,company = metadata.split(",")
            ids_list.append(ids)
            comp_list.append(company)

            tokens = tokenize(tweet)
            X_testset.append(tokens)

    X_testset_to_pred = [" ".join(tokens) for tokens in X_testset]
    testset_vector = tfidf.transform(X_testset_to_pred)
    print("Predicting irrelated and related tweets...")
    pred_svm = svc.predict(testset_vector)

    X_testset_nn = np.array(X_testset_to_pred)
    print("Predicting tags...")
    pred_nn = np.argmax(model.predict(X_testset_nn), axis=1)
    tags_nn = [encod_res[i] for i in pred_nn]

    final_tags = []
    for i in range(len(X_testset)):
        if pred_svm[i] == "irr":
            final_tags.append("irr")
        else:
            final_tags.append(tags_nn[i])

    print("First 15 final tags:", final_tags[0:15])
    print("Starting write the result into file...")
    fw = open("output.txt", "w")
    for i in range(len(X_testset)):
        fw.write(str(ids_list[i])+","+str(final_tags[i])+","+str(comp_list[i])+str(testset_orig[i]))
    print("Done!")

def main():
    tweet = input("Type a tweet : \n")
    tweet_vector = tfidf.transform([tweet])
    prediction = svc.predict(tweet_vector)
    print("Prediction (SVM) : ",prediction[0])
    if prediction[0] == "irr":
        print("This tweet is irrelevant")
    else:
        tweet_to_predict = [" ".join(tokenize(tweet))]
        tweet_to_predict = np.array(tweet_to_predict)
        prediction = np.argmax(model.predict(tweet_to_predict), axis=1)
        print(f"This tweet is {encod_res[prediction[0]]})

if __name__ == "__main__":
    #main()
    testset()
