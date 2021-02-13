import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB 

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

print("Random Forest (40 est)")
evaluate_model(rf)

def find_best_RF(max):
    param_test1 = {'n_estimators':range(10,max,10)}
    gsearch1 = GridSearchCV(
        estimator = RandomForestClassifier(random_state=0), 
        param_grid = param_test1, scoring='f1_macro', cv=3)
    gsearch1.fit(X_train, y_train)
    print(gsearch1.best_params_, gsearch1.best_score_)

# find_best_RF(61) -> best_param:40 trees

#svc = svm.SVC(kernel='rbf')
#print("Fitting SVC Model...")
#svc.fit(X_train, y_train)

#print("SVC model")
#evaluate_model(svc)

def StackingMethod(X,y):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    clf1 = RandomForestClassifier(n_estimators=40, random_state=0)
    clf2 = svm.SVC(kernel='rbf')
    # clf3 = GaussianNB()

    estim_models = [
        ('Random Forest', clf1),
        ('SVM', clf2),
    ]

    sclf = StackingClassifier(estimators=estim_models, final_estimator=clf2)
    print("Training Stacking Classifier...")
    sclf.fit(features_train, target_train)

    y_pred = sclf.predict(features_test)
    print(accuracy_score(y_pred, target_test))
    print(confusion_matrix(target_test, y_pred))
    print(classification_report(target_test, y_pred))

    print("3 Fold Cross Validation:")
    for clf, label in zip([clf1, clf2, sclf], ["Random Forest", "SVM Classifier", "Stacking Classifier"]): 
        scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    return sclf

#StackingMethod(vectors, y)