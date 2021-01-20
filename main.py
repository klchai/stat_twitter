import pandas as pd

def tokenization(tweet):
    tokens=[]
    for word in tweet.split():
        if word[0]=="@":
            word=word[1:]
            tokens.append(word)
        elif word[0]=="#":
            word=word[1:]
            start_index=0
            for i,letter in enumerate(word):
                if letter.isupper():
                    if i!=0:
                        tokens.append(word[start_index:i])
                    start_index=i
                        
                elif i==len(word)-1:
                    tokens.append(word[start_index:])
                else:
                    continue
    return tokens

data=[]

with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,type_tweet,company = metadata.split(",")
        company=company[:-1]
        tokens=tokenization(tweet)
        data.append((type_tweet,company,tokens))

df=pd.DataFrame(data,columns=["Type","Company","Tweet"])

