import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Load model & tokenizer
model_path = r"F:\Internship\finbert_sentiment_model"  # raw string to avoid issues with backslashes
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

#model = AutoModelForSequenceClassification.from_pretrained("./finbert_sentiment_model")
#model = AutoModelForSeqcocccuenceClassification.from_pretrained("./finbert_sentiment_model")
#tokenizer = AutoTokenizer.from_pretrained("./finbert_sentiment_model")

# Label mapping
label_map = {0: "positive", 1: "neutral", 2: "negative"}

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    return label_map[pred]

# Streamlit UI
st.set_page_config(page_title="FinBERT Sentiment Analyzer", layout="centered")
st.title("💼 Financial News Sentiment Analyzer")
st.write("Enter a financial sentence or headline to analyze its sentiment:")

user_input = st.text_area("Input Sentence", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment.upper()}**")
    else:
        st.warning("Please enter a sentence to analyze.")
