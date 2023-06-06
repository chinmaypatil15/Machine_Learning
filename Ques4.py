import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('online_shoppers_intention.csv')

print(data.head())

print(data.shape)

print(data.info())

# Convert target variable to categorical
data['Revenue'] = data['Revenue'].astype(str)

# Extract the relevant features for revenue prediction
features = data.drop(['Revenue'], axis=1)

# Convert weekend column to numerical values (0 for False, 1 for True)
features['Weekend'] = features['Weekend'].astype(int)

# Convert informational duration column to numerical values (0 for False, 1 for True)
features['Informational_Duration'] = features['Informational_Duration'].apply(lambda x: 1 if x > 0 else 0)

# Encode categorical features using one-hot encoding
features = pd.get_dummies(features)

# Extract the target variable (Revenue)
target = data['Revenue']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict the revenue on the test set
y_pred = rf_classifier.predict(X_test)
# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
# Print the accuracy and confusion matrix
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)