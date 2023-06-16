from sentence_transformers import SentenceTransformer
from numpy import savetxt
import pandas as pd
import numpy as np
import torch
import contractions
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')

def remove_contractions(text):
  text_without_contractions = contractions.fix(text)
  return text_without_contractions

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator)
    return text_without_punct

model_checkpoint = 'bert-base-uncased'

df = pd.read_csv("data/movies_data.csv")
# drop other columns
df = df.drop(['Origin/Ethnicity','Director','Cast','Wiki Page'],axis=1)

# prepare the input for the model (Release Year + Genre + Plot)
df['Release Year'] = df['Release Year'].astype(str)
df['input'] = df.apply(lambda row: ' '.join(row[['Release Year', 'Genre', 'Plot']]), axis=1)

# lowercase
df['input']=df['input'].apply(lambda x: x.lower())

# remove contractions (she'll => she will)
df['input'] = df['input'].apply(remove_contractions)

# remove punctuation (?!,)
df['input'] = df['input'].apply(remove_punctuation)

# remove stop words
df['input'] = df['input'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

inputs = df['input'].astype(str).tolist()

df = df.drop(['input','Plot'],axis=1)

df.to_csv("data/movies_data_cleaned.csv",index = False)

model = SentenceTransformer(model_checkpoint)

# save the model
model.save("data/model")

# encoding
sen_embeddings = model.encode(inputs)

# save the embeddings in a csv file
savetxt('data/data.csv', sen_embeddings, delimiter=',')