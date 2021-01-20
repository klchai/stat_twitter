import pandas as pd

def tokenization(tweet):
    tokens=[]
    for word in tweet.split():
        if "." in word:
            if word==".":
                continue
            else:
                sub_words=word.split(".")
                for sub_word in sub_words:
                    if sub_word!="":
                        print("sub word of ",word," ",sub_word)
                        tokens.append(sub_word.lower())
        if word[0]=="@":
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
    return tokens

data=[]

with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet = line[:14],line[14:]
        _,tag,company = metadata.split(",")
        company=company[:-1]
        tokens=tokenization(tweet)
        data.append((tag,company,tokens))

df=pd.DataFrame(data,columns=["Tag","Company","Tweet"])
print(df.head(n=100))