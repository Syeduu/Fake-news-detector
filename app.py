import gradio as gr
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Setup stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    return "🟢 Real News" if pred[0] == 1 else "🔴 Fake News"

# Interface
interface = gr.Interface(
    fn=predict_news,
    inputs="text",
    outputs="text",
    title="📰 Fake News Detector",
    description="Type a news headline or article below to check if it's real or fake."
)

interface.launch()
