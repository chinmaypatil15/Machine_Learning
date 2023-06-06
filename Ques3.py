import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import jaccard, pdist, squareform
import Levenshtein

# Load the news category dataset
df = pd.read_json('News_Category_Dataset_v2.json', lines=True)

# Preprocess the data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

df['preprocessed_headline'] = df['headline'].apply(preprocess_text)
df['preprocessed_short_description'] = df['short_description'].apply(preprocess_text)

# Vectorize the preprocessed text using TF-IDF
vectorizer = TfidfVectorizer()
headline_vectors = vectorizer.fit_transform(df['preprocessed_headline'])
description_vectors = vectorizer.transform(df['preprocessed_short_description'])

# Function to find the most similar data using different similarity algorithms
def find_similar_data(query_vector, vectors, similarity_algorithm):
    similarities = similarity_algorithm(query_vector, vectors)
    most_similar_index = np.argmax(similarities)
    return df.iloc[most_similar_index]

# Example usage: Find the most similar data using different similarity algorithms

# Cosine Similarity
query = "Technology"
query_vector = vectorizer.transform([preprocess_text(query)])
most_similar_cosine = find_similar_data(query_vector, headline_vectors, cosine_similarity)
print("Most similar data using Cosine Similarity:")
print(most_similar_cosine[['headline', 'short_description']])
print()

# Euclidean Distance
query = "Sports"
query_vector = vectorizer.transform([preprocess_text(query)])
most_similar_euclidean = find_similar_data(query_vector, headline_vectors, euclidean_distances)
print("Most similar data using Euclidean Distance:")
print(most_similar_euclidean[['headline', 'short_description']])
print()

# Jaccard Similarity
query = "Politics"
query_vector = vectorizer.transform([preprocess_text(query)])
most_similar_jaccard = find_similar_data(query_vector, headline_vectors, lambda u, v: 1 - jaccard(u.toarray(), v.toarray()))
print("Most similar data using Jaccard Similarity:")
print(most_similar_jaccard[['headline', 'short_description']])
print()

# Levenshtein Distance
query = "Business"
query_vector = preprocess_text(query)
distances = pdist(df['preprocessed_headline'].apply(lambda x: Levenshtein.distance(x, query_vector)).values.reshape(-1, 1), 'euclidean')
similar_index = np.argmin(distances)
most_similar_levenshtein = df.iloc[similar_index]
print("Most similar data using Levenshtein Distance:")
