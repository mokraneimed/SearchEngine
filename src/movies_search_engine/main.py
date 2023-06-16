import torch
import data_cleaner
from search_engine.search_engine import SearchEngine
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')

search_engine = SearchEngine()

while(True):
   k = int(input("Number of movies : "))
   genre = input("Genre : ")
   release_year = input("Release year : ")
   plot = input("Plot : ")

# prepare the input
   input_ = release_year+" "+genre+" "+plot

# clean the input
   input_ = input_.lower()
   input_ = data_cleaner.remove_contractions(input_)
   input_ = data_cleaner.remove_punctuation(input_)
   input_ = ' '.join([word for word in input_.split() if word not in (stop)])
   print(input_)

   print(search_engine.search_movies(input_,k))