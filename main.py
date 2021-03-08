import pandas as pd
import numpy as np

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.utils import np_utils
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
# count_vect = CountVectorizer(min_df=1, ngram_range=(1, 2), stop_words='english')
tfidf = TfidfVectorizer(min_df=1, norm='l2', ngram_range=(1, 2), stop_words='english')

print("Creating Vectors of tweets...")
#features = count_vect.fit_transform(tweets_svm)
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
index_pos, index_neg, index_neu = None, None, None 
i = 0
while (index_pos is None) or (index_neg is None) or (index_neu is None):
    if y_nn[i] == "pos" and index_pos is None:
        index_pos = i
    elif y_nn[i] == "neg" and index_neg is None:
        index_neg = i
    elif y_nn[i] == "neu" and index_neu is None:
        index_neu = i
    i += 1
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
print("Class weight for neg/neu/pos:", weight)

X_train_vote, X_test_vote, y_train_vote, y_test_vote = train_test_split(X_nn, y_nn, test_size=0.2, random_state=0)
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
#model.add(layers.Conv1D(128,3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Fitting Neural Network...")
model.fit(X_train_nn, y_train_nn, epochs=5, class_weight=dict(enumerate(weight)))

loss, accuracy = model.evaluate(X_train_nn, y_train_nn, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_nn, y_test_nn, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# Voting Classifier
vote_tfidf = TfidfVectorizer(min_df=1, norm='l2', ngram_range=(1, 2), stop_words='english')
vote_tfidf.fit_transform(X_nn)
X_train_vote = vote_tfidf.transform(X_train_vote)
X_test_vote = vote_tfidf.transform(X_test_vote)

clf1 = LogisticRegression(multi_class='multinomial', random_state=1, class_weight='balanced')
clf2 = RandomForestClassifier(n_estimators=40, random_state=1, class_weight='balanced')
clf3 = MultinomialNB()
vote_soft = VotingClassifier(estimators=[('LR', clf1),('RF',clf2),('Bayes',clf3)], voting='soft')

print("Fitting Voting Classifier...")
vote_soft.fit(X_train_vote, y_train_vote)
vote_pred_y = vote_soft.predict(X_test_vote)
print(classification_report(y_test_vote, vote_pred_y))

def main():
    tweet = input("Type a tweet : \n")
    tweet_vector = tfidf.transform([tweet])
    prediction = svc.predict(tweet_vector)
    print("Prediction (SVM) : ",prediction[0])
    if prediction[0] == "irr":
        print("this tweet is irrelevant")
    else:
        tweet_to_predict = [" ".join(tokenize(tweet))]
        tweet_to_predict = np.array(tweet_to_predict)
        prediction = np.argmax(model.predict(tweet_to_predict), axis=1)
        print("Prediction (NN) : ", encod_res[prediction[0]])
        vect_tweet_to_predict = vote_tfidf.transform(tweet_to_predict)
        pred_vote_soft = vote_soft.predict(vect_tweet_to_predict)
        print("Prediction (Voting) : ", encod_res[pred_vote_soft[0]])

if __name__ == "__main__":
    main()
