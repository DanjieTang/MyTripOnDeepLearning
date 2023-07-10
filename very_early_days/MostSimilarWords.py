import torch
import torch.nn as nn
from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=300)
cos_similarity=nn.CosineSimilarity(dim=0)

read=open("test.txt", encoding="latin1")
lines=read.read().split("\n")
total=len(lines)
correct_counter=0

for element in lines:
    if(element==""):
        continue
    else:
        word_by_word=element.split(" ")
        #Get the embedding of the first word
        word_embedding=glove[word_by_word[0]]

        #Get the embedding of the 2 words in the multiple choice
        choice1_embedding=glove[word_by_word[2]]
        choice2_embedding=glove[word_by_word[3]]

        #Get the cosine similarity of the two embeddings with respect to the word
        cos1=cos_similarity(word_embedding, choice1_embedding)
        cos2=cos_similarity(word_embedding, choice2_embedding)

        if(cos1>cos2):
            #Thinks word 1 is the closer word
            if(word_by_word[1]==word_by_word[2]):
                correct_counter+=1
        else:
            #Thinks word 2 is the closer word
            if(word_by_word[1]==word_by_word[3]):
                correct_counter+=1


print(str(float(correct_counter)/total*100)+"% of the guesses were correct")