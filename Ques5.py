# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import folium

pd.options.display.max_rows = None
pd.options.display.max_columns = None
df=pd.read_csv(r"rideshare_kaggle.csv")
print(df.head())

print(df.info())

df['datetime']=pd.to_datetime(df['datetime'])
print(df.columns)

print(df.isnull().sum().sum())
df.dropna(axis=0,inplace=True)
print(df.isnull().sum().sum())
print(df['visibility'].head())

print(df['visibility.1'].head())
print(df.drop(['visibility.1'],axis=1))

pd.set_option('display.max_rows', 72)
print(df.groupby(by=["source","destination"]).price.agg(["mean"]))


# Step 2: Unsupervised Learning - High Booking Area Prediction

# Use K-means clustering to identify high booking areas
X = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
df['cluster_label'] = kmeans.labels_

# Step 3: Supervised Learning - Price Prediction

# Prepare data for supervised learning
X_train = df[['latitude', 'longitude']]
y_train = df['price']

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Map Visualization

# Create a map centered around a specific location
map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)

# Add markers for high booking areas
for index, row in df.iterrows():
    if row['cluster_label'] == 1:  # Change the condition based on your cluster label for high booking areas
        folium.Marker([row['latitude'], row['longitude']],
                      #popup=f"Booking Frequency: {row['booking_frequency']}",
                      icon=folium.Icon(color='red')).add_to(map)

# Display the map
map.save('booking_areas.html')
