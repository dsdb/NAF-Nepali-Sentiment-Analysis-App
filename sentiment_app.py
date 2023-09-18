import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "dpkrm/NepaliSentimentAnalysis"

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

@st.cache_data
def perform_sentiment_analysis(text):
    # Tokenize the input text and perform inference
    inputs = get_tokenizer()(text, return_tensors="pt", truncation=True, padding=True)
    outputs = get_model()(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Streamlit app
st.title("Nepali Sentiment Analysis App")

# Input text box
text = st.text_area("Enter text for sentiment analysis")

if st.button("Analyze"):
    if text:
        # Perform sentiment analysis
        sentiment_label = perform_sentiment_analysis(text)

        # Map the label to the corresponding sentiment
        sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        sentiment_value = sentiment_mapping[sentiment_label]

        # Display the result
        st.write(f"Sentiment: {sentiment_value}")
    else:
        st.warning("Please enter some text for analysis")
