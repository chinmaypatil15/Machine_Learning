import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify

data_1 = pd.read_csv(r'archiveQ7/data.csv')
data_2 = pd.read_csv(r'archiveQ7/data_2genre.csv')

data = pd.concat([data_1, data_2])

## Checking top 5 row of Dataset
print(data.head())

print(data['label'].value_counts())

data['label'] = data['label'].replace(to_replace={1: 'pop', 2: 'classical'})

print(data['label'].value_counts())

data['label'] = data['label'].replace(to_replace={1: 'pop', 2: 'classical'})

print(data['label'].value_counts())

plt.figure(figsize=(30,10))

sns.kdeplot(data=data.loc[data['label']=='jazz', 'tempo'], label="Jazz")
sns.kdeplot(data=data.loc[data['label']=='pop', 'tempo'], label="Pop")
sns.kdeplot(data=data.loc[data['label']=='classical', 'tempo'], label="Classical")
sns.kdeplot(data=data.loc[data['label']=='hiphop', 'tempo'], label="Hiphop")
sns.kdeplot(data=data.loc[data['label']=='disco', 'tempo'], label="Disco")
sns.kdeplot(data=data.loc[data['label']=='country', 'tempo'], label="Country")
sns.kdeplot(data=data.loc[data['label']=='rock', 'tempo'], label="Rock")
sns.kdeplot(data=data.loc[data['label']=='metal', 'tempo'], label="Metal")
sns.kdeplot(data=data.loc[data['label']=='reggae', 'tempo'], label="Reggae")
sns.kdeplot(data=data.loc[data['label']=='blues', 'tempo'], label="Blues")

print(plt.title("Distribution of tempos by genre", fontsize = 18))

print(plt.xlabel("Tempo", fontsize = 18))

print(plt.legend())

print(plt.figure(figsize=(30,10)))

genres = data['label'].unique()

tempos = [ data[data['label']==x].tempo.mean() for x in genres ]

sns.barplot(x=genres, y=tempos, palette="deep")

print(plt.title("Average tempo by genre", fontsize = 18))

print(plt.xlabel('Genre', fontsize = 18))
print(plt.ylabel('Mean Tempo', fontsize = 18))


# Select relevant features
features = data.drop(['filename', 'label'], axis=1)

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(normalized_features)

# Add the predicted clusters to the dataset
data['cluster'] = clusters

# Flask application setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get the features for the new sample
    sample_features = [data['tempo'], data['energy'], data['danceability']]

    # Normalize the features
    normalized_sample_features = scaler.transform([sample_features])

    # Predict the cluster for the new sample
    cluster = kmeans.predict(normalized_sample_features)[0]

    # Get the majority genre of the cluster
    majority_genre = data[data_1['cluster'] == cluster]['label'].mode()[0]

    return jsonify({'genre': majority_genre})
