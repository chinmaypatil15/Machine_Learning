import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')
print(data.head())

print(data.shape)

print(data.info())

print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Split the data into question pairs and labels
questions = data[['question1', 'question2']]
labels = data['is_duplicate']

questions_train, questions_test, labels_train, labels_test = train_test_split(questions, labels, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer()
tfidf_train = tfidf.fit_transform(questions_train['question1'] + ' ' + questions_train['question2'])

model = LogisticRegression()
model.fit(tfidf_train, labels_train)

tfidf_test = tfidf.transform(questions_test['question1'] + ' ' + questions_test['question2'])
predictions = model.predict(tfidf_test)

accuracy = accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy)