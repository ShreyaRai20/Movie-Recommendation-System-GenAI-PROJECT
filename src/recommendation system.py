#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Load the datasets
credits = pd.read_csv('data/tmdb_5000_credits.csv')
movies = pd.read_csv('data/tmdb_5000_movies.csv')

# Preprocessing
movies['overview'] = movies['overview'].fillna("")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode movie overviews using BERT
def encode_overview(overview):
    inputs = tokenizer(overview, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

movies['bert_embedding'] = movies['overview'].apply(encode_overview)

# Compute cosine similarity matrix
bert_embeddings = np.stack(movies['bert_embedding'].values)
cosine_sim = cosine_similarity(bert_embeddings, bert_embeddings)

# Create indices for movie titles
indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    sim_index = [i[0] for i in sim_scores]
    return movies['original_title'].iloc[sim_index]

# Test the recommendation system
print(get_recommendations('The Dark Knight Rises'))
print(get_recommendations('The Matrix'))
