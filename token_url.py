def tokenization(tweet):
    tokens=[]
    #ponctuation=[".",";","!",",",":","-"]
    ponctuation=[".",";","!",",","-","\n"]
    words = tweet.split(" ")
    for i in words:
        if "http" in i:
            # tokens.append(i.lower())
            words.remove(i)

    sentences = ' '.join(words)
            
    for p in ponctuation:
        sentences=sentences.replace(p," ")
    for word in sentences.split():
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
