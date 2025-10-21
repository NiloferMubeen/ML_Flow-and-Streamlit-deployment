import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the trained pipeline
with open("best_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("Tip Prediction App")  # Example if your target is tip

# User inputs
total_bill = st.number_input("Total Bill ($)", min_value=0.0, step=0.1)
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day of Week", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])
size = st.number_input("Table Size", min_value=1, step=1)

# Create a DataFrame for the pipeline
input_df = pd.DataFrame({
    "total_bill": [total_bill],
    "sex": [sex],
    "smoker": [smoker],
    "day": [day],
    "time": [time],
    "size": [size]
})

# Predict when button clicked
if st.button("Predict"):
    prediction = pipeline.predict(input_df)
    st.success(f"Predicted Tip: ${prediction[0]:.2f}")

