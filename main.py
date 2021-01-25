import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def tokenization(tweet):
    tokens=[]
    ponctuation=[".",";","!",",",":","-"]
    for p in ponctuation:
        tweet=tweet.replace(p," ")
    for word in tweet.split():
        if word.isdigit():
            continue
        else:
            contains_digit=any(map(str.isdigit,word))
            if contains_digit:
                continue
            elif word[0]=="@":
                word=word[1:]
                tokens.append(word.lower())
            elif word[0]=="#":
                word=word[1:]
                start_index=0
                for i,letter in enumerate(word):
                    if letter.isupper():
                        if i!=0:
                            tokens.append(word[start_index:i].lower())
                        start_index=i
                            
                    elif i==len(word)-1:
                        tokens.append(word[start_index:].lower())
                    else:
                        continue
            else:
                tokens.append(word.lower())
    return tokens

X=[]
y=[]
with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,tag,company = metadata.split(",")
        company=company[:-1]
        tokens=tokenization(tweet)
        X.append(tokens)
        if tag=="pos":
            m=1
        elif tag=="neg":
            m=-1
        else:
            m=0
        y.append(m)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


""" #test
df=pd.DataFrame(data,columns=["Tag","Company","Tweet"])
for i,r in df.iterrows():
    print(r["Tweet"])
"""