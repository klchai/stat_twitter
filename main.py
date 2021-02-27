import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB 
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer 

all_words = set()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenize(tweet):
    tokens=[]
    ponctuation=[".",";","!",",","-","\n"]
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
                            all_words.add(new_word)
                            tokens.append(new_word)
                        start_index=i
                            
                    elif i==len(word)-1:
                        new_word = lemmatizer.lemmatize(word[start_index:].lower())
                        all_words.add(new_word)
                        tokens.append(new_word)
                    else:
                        continue
            else:
                new_word = lemmatizer.lemmatize(word.lower())
                all_words.add(new_word)
                tokens.append(new_word)
    return tokens

X=[]
y=[]
with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,tag,_ = metadata.split(",")
        tokens = tokenize(tweet)
        if tag!="irr":
            X.append(tokens)
            y.append(tag)

print("All tweets are loaded, creating the vectors of tweets...")

def custom_standardize():
    tweets = []
    max_sequence_length = 0
    for tokens in X:
        current_tweet_length = len(tokens)
        if current_tweet_length > max_sequence_length:
            max_sequence_length = current_tweet_length
        tweets.append(" ".join(tokens))
    return tweets,max_sequence_length

tweets,max_sequence_length = custom_standardize()
tweets = np.array(tweets)
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
X_train, X_test, y_train, y_test = train_test_split(tweets,y,test_size=0.2,random_state=0)

MAX_TOKENS_NUM = len(all_words)
EMBEDDING_DIMS = 10

vectorize_layer = TextVectorization(
  max_tokens=MAX_TOKENS_NUM,
  output_mode='int',
  output_sequence_length=max_sequence_length
)
vectorize_layer.adapt(tweets)

model = Sequential()
model.add(Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(layers.Embedding(MAX_TOKENS_NUM,EMBEDDING_DIMS))
model.add(layers.Dense(1, activation="relu"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50)

score = model.evaluate(X_test,y_test)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])