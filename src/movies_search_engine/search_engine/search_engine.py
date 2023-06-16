from sentence_transformers import SentenceTransformer
from numpy import loadtxt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SearchEngine:
    
# prepare the model
    def __init__(self):
# load the model
        self.model = SentenceTransformer("data/model")

# load the dataset
        self.df = pd.read_csv("data/movies_data_cleaned.csv")

# load the encoded dataset
        self.data = loadtxt("data/data.csv", delimiter=',')

    def search_movies(self,input_,k):
# incode the input
        embeddings = self.model.encode(input_)

# calculate the cosine similarity
        scores = cosine_similarity(
            [embeddings],
            self.data
        )

# extract the indexes of the top k movies
        x = np.argsort(scores)
        y = x[0][::-1]
        z = y[0:k]

# extract the top k movies
        movies = []
        for row in z:
            movie = {
                "Title": self.df.iloc[row]["Title"],
                "year": self.df.iloc[row]["Release Year"],
                "genre": self.df.iloc[row]["Genre"]
            }
            movies.append(movie)
        return movies
            