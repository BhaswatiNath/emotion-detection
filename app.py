import streamlit as st
import joblib

# Load the model
model = joblib.load('model.pkl')

st.title("Emotion Detection App")

# User input
user_input = st.text_area("Enter your text:")

if st.button("Predict Emotion"):
    prediction = model.predict([user_input])
    st.write(f"Predicted Emotion: {prediction[0]}")
