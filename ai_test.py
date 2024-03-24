import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('spam_ham_dataset.csv')

print(df.head())
print(df.info())

X = df['text']  
y = df['label']  
y = np.where(y == 'spam', 1, 0)  

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = SVC(kernel='linear', probability=True)
model.fit(X_vectorized, y)
print("trained")

joblib.dump(model, 'spam_classifier_model3.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model and vectorizer saved successfully.")

X_new = [
    "Subject: You've won a prize! Click here to claim it now!",
    "Subject: Meeting reminder: Tomorrow at 10 AM in conference room",
    "Subject: Urgent: Your account needs attention. Please log in.",
    "GIVE ME ALL YOUR MONEY PLEASE!!!!!"
]

X_new_vectorized = vectorizer.transform(X_new)

predictions = model.predict(X_new_vectorized)

for prediction in predictions:
    if prediction == 1:
        print("Spam")
    else:
        print("Ham")
