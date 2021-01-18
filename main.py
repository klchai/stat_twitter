import pandas as pd

data=[]

with open("./train.txt","r") as file:
    for line in file:
        metadata,tweet=line[:14],line[14:]
        data.append((metadata,tweet.split()))

df=pd.DataFrame(data,columns=["Metadata","Tweet"])
print(df.head(n=10))

