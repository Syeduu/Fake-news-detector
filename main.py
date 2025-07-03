import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import joblib

nltk.download('stopwords')

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine title + text (if available), fill if text is missing
fake["text"] = fake["title"].fillna('') + " " + fake["text"].fillna('')
real["text"] = real["title"].fillna('') + " " + real["text"].fillna('')

# Combine and shuffle
df = pd.concat([fake, real], axis=0).dropna(subset=["text"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove duplicates
df = df.drop_duplicates(subset="text")

# Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

df['clean_text'] = df['text'].apply(clean_text)

# Features & labels
X = df['clean_text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc * 100:.2f}%")

# Predict function
import re

def predict_news(news_text):
    cleaned = clean_text(news_text)
    
    # Reject if too short or mostly symbols
    if len(cleaned.split()) < 4 or not re.search(r"[a-zA-Z]{3,}", cleaned):
        return "⚠️ Please enter valid news content"
    
    # Predict normally
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    return "🟢 Real News" if pred == 1 else "🔴 Fake News"


# CLI test
while True:
    user_input = input("\nEnter a news headline (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Prediction:", predict_news(user_input))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
