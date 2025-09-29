import streamlit as st
import pandas as pd
import joblib   # safer than pickle for sklearn models

# -------------------------------
# Load the trained model
# -------------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"❌ Could not load model.pkl: {e}")
    st.stop()

# -------------------------------
# App Title
# -------------------------------
st.title("👟 Shoe Size Prediction App")

st.write("""
This app predicts **Shoe Size** based on:  
- Age  
- Height (cm)  
- Gender  
""")

# -------------------------------
# Load dataset (optional preview)
# -------------------------------
try:
    data = pd.read_csv("shoes_size_age_height_gender_size.csv")
    if st.checkbox("Show sample dataset"):
        st.write(data.head())
except FileNotFoundError:
    st.warning("⚠️ shoes_size_age_height_gender_size.csv not found in the app folder.")

# -------------------------------
# User Inputs
# -------------------------------
st.sidebar.header("Enter Features")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=20)
height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Convert gender into numeric if model was trained that way
# (Assuming Male=1, Female=0)
gender_encoded = 1 if gender == "Male" else 0

# Prepare input for model
features = pd.DataFrame(
    [[age, height, gender_encoded]],
    columns=["age", "height", "gender"]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Shoe Size 👟"):
    try:
        prediction = model.predict(features)
        st.success(f"👟 Predicted Shoe Size: **{prediction[0]:.1f}**")
    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")
