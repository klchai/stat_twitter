import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def tokenization(tweet):
    tokens=[]
    ponctuation=[".",";","!",",","-","\n"]
    useless_words=["i","you","he","she","we","they","it","is","was","to","for"]
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
            elif word.lower() in useless_words:
                continue
            elif word[0]=="@":
                word=word[1:].lower()
                dico.add(word)
                tokens.append(word)
            elif word[0]=="#":
                word=word[1:]
                start_index=0
                for i,letter in enumerate(word):
                    if letter.isupper():
                        if i!=0:
                            dico.add(word[start_index:i].lower())
                            tokens.append(word[start_index:i].lower())
                        start_index=i
                            
                    elif i==len(word)-1:
                        dico.add(word[start_index:].lower())
                        tokens.append(word[start_index:].lower())
                    else:
                        continue
            else:
                tokens.append(word.lower())
    return tokens

dico=set()
X=[]
y=[]
with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,tag,company = metadata.split(",")
        company=company[:-1]
        tokens=tokenization(tweet)
        X.append(tokens)
        y.append(tag)

vectors=[]
for tweet in X:
    vector=[1 if word in tweet else 0 for word in dico]
    vectors.append(vector)


X_train, X_test, y_train, y_test = train_test_split(vectors,y,test_size=0.2,random_state=0)
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))