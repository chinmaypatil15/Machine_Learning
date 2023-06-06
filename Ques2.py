
## Import the necessary libraries:-
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = arff.loadarff('ObesityDataSet_raw_and_data_sinthetic.arff')
df = pd.DataFrame(data[0])

print(df.head())

print(df.shape)

print(df.info())

print(df.columns)

print(df.describe())

print(df.info())

print(df.columns)

# Preprocess the dataset
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['family_history_with_overweight'] = encoder.fit_transform(df['family_history_with_overweight'])
df['FAVC'] = encoder.fit_transform(df['FAVC'])
df['CAEC'] = encoder.fit_transform(df['CAEC'])
df['SMOKE'] = encoder.fit_transform(df['SMOKE'])
df['SCC'] = encoder.fit_transform(df['SCC'])
df['CALC'] = encoder.fit_transform(df['CALC'])
df['MTRANS'] = encoder.fit_transform(df['MTRANS'])
df['NObeyesdad'] = encoder.fit_transform(df['NObeyesdad'])

print(df.info())

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_predictions = dt_clf.predict(X_test)

# Logistic Regression Classifier
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_predictions = lr_clf.predict(X_test)

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_predictions = rf_clf.predict(X_test)

# Support Vector Machine (SVM) Classifier
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_predictions = svm_clf.predict(X_test)

# Print classification reports for each classifier
print("Decision Tree Classifier:")
print(classification_report(y_test, dt_predictions))

# Print classification reports for each classifier
print("Logistic Regression Classifier:")
print(classification_report(y_test, lr_predictions))

print("Random Forest Classifier:")
print(classification_report(y_test, rf_predictions))

print("SVM Classifier:")
print(classification_report(y_test, svm_predictions))

# Train and evaluate the models
# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
logistic_regression_predictions = logistic_regression.predict(X_test)
print("Logistic Regression:")
#print(confusion_matrix(y_test, logistic_regression_predictions))
print(classification_report(y_test, logistic_regression_predictions))

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
decision_tree_predictions = decision_tree.predict(X_test)
print("Decision Tree:")
#print(confusion_matrix(y_test, decision_tree_predictions))
print(classification_report(y_test, decision_tree_predictions))

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
random_forest_predictions = random_forest.predict(X_test)
print("Random Forest:")
#print(confusion_matrix(y_test, random_forest_predictions))
print(classification_report(y_test, random_forest_predictions))

# Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
print("Support Vector Machine:")
#print(confusion_matrix(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))
