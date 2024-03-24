import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


loaded_model = joblib.load('spam_classifier_model3.joblib')

vectorizer = joblib.load('vectorizer.joblib')

def classify_email(email_text):
    email_text_vectorized = vectorizer.transform([email_text])
    
    probabilities = loaded_model.predict_proba(email_text_vectorized)[0]
    
    prediction = loaded_model.predict(email_text_vectorized)[0]
    
    return prediction, probabilities

def main():
    email_text = input("Enter the email text: ")
    
    classification, probabilities = classify_email(email_text)
    
    
    print(f"Probability of spam: {probabilities[1]*100:.2f}%")
    print(f"Probability of not spam: {probabilities[0]*100:.2f}%")

if __name__ == "__main__":
    main()
