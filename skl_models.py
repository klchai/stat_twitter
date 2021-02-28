import pandas as pd
import numpy as np

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

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
        X.append(tokens)
        y.append(tag)
        """
        if tag!="irr":
            X.append(tokens)
            y.append("nrr")
        else:
            X.append(tokens)
            y.append(tag)
        """

print("All tweets are loaded.")

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
#y = le.transform(y)
#print("Labels:", y)

# Vectorizer 
vectorizer = CountVectorizer(min_df=5)
print("Creating Vectors of tweets...")
features = vectorizer.fit_transform(tweets).toarray()
print("Vectors shape:",features.shape)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=0)
print("X_train:{}, y_train:{}".format(X_train[0], y_train[0]))

def evaluate_model(model, show_detail=False):
    y_pred = model.predict(X_test)
    if show_detail:
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    ev = cross_val_score(model, X_test, y_test, cv=3)
    print("3 Fold Cross Validation, accuracy:",ev.mean())

rf = RandomForestClassifier(n_estimators=40, random_state=0)
rf.fit(X_train, y_train)
print("Fitting RandomForest Classifier (40 est)...")
evaluate_model(rf, show_detail=True)

svc = svm.SVC(kernel='rbf')
print("Fitting SVM Classifier...")
svc.fit(X_train, y_train)
evaluate_model(svc, show_detail=True)

bayes = GaussianNB()
print("Fitting Gaussian NB...")
bayes.fit(X_train, y_train)
evaluate_model(bayes, show_detail=True)