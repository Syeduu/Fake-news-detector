import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

# Predict function with validation
def predict_news(text):
    cleaned = clean_text(text)
    
    if len(cleaned.split()) < 4 or not re.search(r"[a-zA-Z]{3,}", cleaned):
        return None  # Invalid input
    
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    return "🟢 Real News" if pred == 1 else "🔴 Fake News"

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")
st.subheader("Check if a news article or headline is Real or Fake")

user_input = st.text_area("Enter the news text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_news(user_input)
        if result is None:
            st.error("⚠️ Please enter valid news content (not just random symbols or short text).")
        else:
            st.success(f"Prediction: {result}")
