import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("assets/model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the saved encoder
with open("assets/label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

st.set_page_config(page_title="Shoe Size Prediction", page_icon="👟", layout="centered")

st.title("👟 Shoe Size Prediction App")
st.markdown("Enter the details below to predict the *Shoe Size* using a trained Linear Regression model.")

# Sidebar info
st.sidebar.header("ℹ️ About")
st.sidebar.info("This app uses a *Linear Regression* model trained with scikit-learn. "
                "It predicts the shoe size based on Age, Height, and Gender.")

# Input fields
age = st.number_input("Age", min_value=10, max_value=70, value=25)
height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)
gender = st.selectbox("Gender", encoder.classes_)  # Use classes from encoder

# Encode gender using the saved encoder
gender_encoded = encoder.transform([gender])[0]

# Prepare feature vector
features = np.array([[age, height, gender_encoded]])

if st.button("Predict Shoe Size"):
    prediction = model.predict(features)
    st.success(f"Predicted Shoe Size: *{prediction[0]:.2f}*")