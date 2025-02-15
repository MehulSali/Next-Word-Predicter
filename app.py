import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model("next_word_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Title
st.title("Next Word Predictor")

# Load the dataset
def load_data():
    data = pd.read_csv("Quotes2.csv")
    data = data.drop_duplicates().reset_index(drop=True)
    new_data = data['Quote']
    return new_data

data = load_data()

# Display dataset preview
if st.checkbox("Show dataset preview"):
    st.write(data.sample(5))

# User input
user_input = st.text_input("Enter the beginning of a sentence:", "")

# Function to predict the next word(s)
def predict_next_word(text, num_words=10):
    if not text.strip():
        raise ValueError("Input text cannot be empty!")
    for _ in range(num_words):
        tokenized_text = tokenizer.texts_to_sequences([text])[0]
        padded_text = pad_sequences([tokenized_text], maxlen=107, padding='pre')
        predicted_pos = np.argmax(model.predict(padded_text, verbose=0))
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_pos:
                text += " " + word
                break
    return text

# Prediction logic
if st.button("Predict"):
    try:
        with st.spinner("Predicting the next words..."):
            predicted_sentence = predict_next_word(user_input, num_words=10)
        st.success("Prediction completed!")
        st.write(f"**Predicted sentence:** {predicted_sentence}")
    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
