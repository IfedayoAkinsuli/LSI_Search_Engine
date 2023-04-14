query = input("Please Enter your search")

import numpy as np
import requests
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer



url_list = [
"https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher's%20Stone.txt",
"https://raw.githubusercontent.com/ganesh-k13/shell/master/test_search/www.glozman.com/TextPages/01%20-%20The%20Fellowship%20Of%20The%20Ring.txt",
"https://raw.githubusercontent.com/laumann/ds/master/hashing/books/jane-austen-pride-prejudice.txt"
]

documents = ["Harry Potter and the Philosopher's Stone","Fellowship of the Ring","Pride and Prejudice"]
text = []

for _,i in enumerate(url_list):
    req = requests.get(i)
    text.append(req.text)

def preprocessing(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)

    lower_filter = [w.lower() for w in text]
    filtered_text = []

    filtered_text = [i for i in filtered_text if not i.isdigit()]

    stop_words = stopwords.words('english') + ['j','page','k','said','rowling','quot','back','mr','mrs']

    for words in lower_filter:
        if words not in stop_words:
            filtered_text.append(words)

    filtered_text = [i for i in filtered_text if not i.isdigit()]

    lem = WordNetLemmatizer()

    filtered_text = [lem.lemmatize(w) for w in filtered_text]

    return filtered_text


def query_processing(text):
    tokenizer = RegexpTokenizer(r'\w+')
    processed_query = tokenizer.tokenize(text)

    processed_query = [w.lower() for w in processed_query]

    return processed_query


def doc_ranking_score(similarities):
    for i in similarities:
        t = np.argsort(i)

    return t[0]


processed = [preprocessing(i) for _,i in enumerate(text)]

from gensim.corpora import Dictionary
from gensim.models import LsiModel

X = Dictionary(processed)
doc_term_matrix = [X.doc2bow(doc) for doc in processed]
lsi_model = LsiModel(corpus=doc_term_matrix,num_topics=3,id2word=X)

processed_query = query_processing(query)

vector_query = lsi_model[Dictionary(processed).doc2bow(processed_query)]

from gensim import similarities
index = similarities.MatrixSimilarity(lsi_model[doc_term_matrix])

similarities = index[vector_query]

rel = doc_ranking_score(similarities)

print(f'{documents[rel]}')
