import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels: Fake = 0, Real = 1
fake["label"] = 0
real["label"] = 1

# Combine datasets
df = pd.concat([fake, real], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle rows

# Drop rows with missing text
df = df.dropna(subset=["text"])

# Clean text - remove punctuation, lowercase, remove stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

df['clean_text'] = df['text'].apply(clean_text)

print("✅ Text cleaned")
print("Example cleaned text:\n", df['clean_text'].iloc[0][:500])  # show first 500 chars
# Features (X) and Labels (y)
X = df['clean_text']
y = df['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_df=0.7)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Show data info
print("✅ Dataset Loaded")
print("Total entries:", df.shape[0])
print("Sample data:\n", df[['title', 'label']].head())

# Train a Passive Aggressive Classifier (great for fake news detection)
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc * 100:.2f}%")


# Function to predict new headlines
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)
    return "🟢 Real News" if pred[0] == 1 else "🔴 Fake News"

# Example usage
while True:
    user_input = input("\nEnter a news headline (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result = predict_news(user_input)
    print("Prediction:", result)


import joblib
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
