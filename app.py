import streamlit as st
import joblib

# Load the model
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Emotion Detection App")

# User input
user_input = st.text_input("Enter your text:")

if user_input:
    # Transform user input
    processed_input = vectorizer.transform([user_input])
    # Make prediction
    prediction = model.predict(processed_input)
    st.write(f"Predicted sentiment: {prediction[0]}")

