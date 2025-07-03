import gradio as gr
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    
    # Validate input
    if len(cleaned.split()) < 4 or not re.search(r"[a-zA-Z]{3,}", cleaned):
        return "⚠️ Please enter valid news content"
    
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    return "🟢 Real News" if pred[0] == 1 else "🔴 Fake News"

# Gradio Interface
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=6, placeholder="Enter news headline or article..."),
    outputs="text",
    title="📰 Fake News Detector",
    description="Enter a news article or headline to check if it's Real or Fake."
)

interface.launch()
