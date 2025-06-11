import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Title
st.title("🌸 Iris Flower Classifier")

# Inputs
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    # Map prediction to label
    labels = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Flower: {labels[prediction]}")
