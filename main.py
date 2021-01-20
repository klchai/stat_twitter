import pandas as pd

def tokenization(tweet):
    tokens=[]
    ponctuation=[".",";","!",",",":","-"]
    for p in ponctuation:
        tweet=tweet.replace(p," ")
    for word in tweet.split():
        if word.isdigit():
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

data=[]

with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,tag,company = metadata.split(",")
        company=company[:-1]
        tokens=tokenization(tweet)
        data.append((tag,company,tokens))

""" #test
df=pd.DataFrame(data,columns=["Tag","Company","Tweet"])
for i,r in df.iterrows():
    print(r["Tweet"])
"""