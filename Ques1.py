import pandas as pd
import re  # Regular expression library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the dataset
data = pd.read_csv('instagram_reach.csv')

# Explore the dataset
print(data.head())  # Print the first few rows of the dataset
print(data.info())  # Get information about the dataset

# Handle missing values
data = data.dropna()  # Drop rows with missing values

# Split the dataset into input features and target variables
X = data.drop(['Likes', 'Time_since_posted'], axis=1)  # Input features
y_likes = data['Likes']  # Target variable - Number of likes
y_time = data['Time_since_posted']  # Target variable - Time since posted

# Extract numeric values from 'Time_since_posted' using regular expressions
y_time = y_time.str.extract(r'(\d+)').astype(int)  # Extract numeric part and convert to int

# Perform one-hot encoding for categorical features
categorical_features = ['USERNAME', 'Caption', 'Hashtags']
preprocessor = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_features)],
    remainder='passthrough'
)
X_encoded = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_likes_train, y_likes_test, y_time_train, y_time_test = train_test_split(
    X_encoded, y_likes, y_time, test_size=0.2, random_state=42
)

# Initialize the model
model_likes = LinearRegression()  # Example: Linear regression for predicting likes
model_time = RandomForestRegressor()  # Example: Random forest regression for predicting time

# Fit the model on the training data
model_likes.fit(X_train, y_likes_train)
model_time.fit(X_train, y_time_train)

# Make predictions on the testing data
likes_predictions = model_likes.predict(X_test)
time_predictions = model_time.predict(X_test)

# Evaluate the predictions
likes_mse = mean_squared_error(y_likes_test, likes_predictions)
likes_mae = mean_absolute_error(y_likes_test, likes_predictions)
time_mae = mean_absolute_error(y_time_test, time_predictions)

print("Likes MSE:", likes_mse)
print("Likes MAE:", likes_mae)
print("Time MAE:", time_mae)
