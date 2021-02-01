import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV

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
rf1 = RandomForestClassifier(n_estimators=20, random_state=0)
rf1.fit(X_train, y_train)
rf1_y_pred = rf1.predict(X_test)

print("Random Forest 1 (20 est)")
print(confusion_matrix(y_test,rf1_y_pred))
print(classification_report(y_test,rf1_y_pred))
print("Accuracy:", accuracy_score(y_test, rf1_y_pred))

rf2 = RandomForestClassifier(n_estimators=40, random_state=0)
rf2.fit(X_train, y_train)
rf2_y_pred = rf2.predict(X_test)

print("Random Forest 2 (40 est)")
print(confusion_matrix(y_test,rf2_y_pred))
print(classification_report(y_test,rf2_y_pred))
print("Accuracy:", accuracy_score(y_test, rf2_y_pred))

def find_best_RF(max):
    param_test1 = {'n_estimators':range(10,max,10)}
    gsearch1 = GridSearchCV(
        estimator = RandomForestClassifier(
            random_state=0
        ), param_grid = param_test1, scoring='f1_macro',cv=3)
    gsearch1.fit(X_train, y_train)
    print(gsearch1.best_params_, gsearch1.best_score_)

# find_best_RF(61)

svc = svm.SVC(kernel='rbf')
print("Fitting SVC Model...")
svc.fit(X_train, y_train)
svc_y_pred = svc.predict(X_test)

print(confusion_matrix(y_test, svc_y_pred))
print(classification_report(y_test, svc_y_pred))
print("Accuracy:", accuracy_score(y_test, svc_y_pred))

def evaluate_cv(model1, model2, model3):
    cv_RF1 = cross_val_score(model1, X_test, y_test, cv=10)
    cv_RF2 = cross_val_score(model2, X_test, y_test, cv=10)
    cv_SVC = cross_val_score(model3, X_test, y_test, cv=10)
    print("RF1: {}, RF2: {}, SVM: {}".format(cv_RF1.mean(), cv_RF2.mean(), cv_SVC.mean()))

print("Evaluation of models:")
evaluate_cv(rf1, rf2, svc)