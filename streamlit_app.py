import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")
st.subheader("Check if a news headline or article is Real or Fake")

# Input text
user_input = st.text_area("Enter the news text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        result = "🟢 Real News" if pred[0] == 1 else "🔴 Fake News"
        st.success(f"Prediction: {result}")
