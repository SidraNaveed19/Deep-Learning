import streamlit as st
import pickle
import numpy as np

# Load model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

# Inputs
age = st.number_input("Age", 1, 120)
sex = st.number_input("Sex (0 = Female, 1 = Male)")
cp = st.number_input("Chest Pain Type (0–3)")
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol Level", 100, 600)
fbs = st.number_input("Fasting Blood Sugar ")
restecg = st.number_input("Rest ECG (0–2)")
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.number_input("Exercise Induced Angina ")
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0)
slope = st.number_input("Slope (0–2)")
ca = st.number_input("Number of Major Vessels (0–3)")
thal = st.number_input("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible)")

# Prepare input
import pandas as pd

features = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]],
                        columns=['age', 'sex', 'cp', 'trestbps', 'chol',
                                 'fbs', 'restecg', 'thalach', 'exang',
                                 'oldpeak', 'slope', 'ca', 'thal'])
# Scale input
features = scaler.transform(features)

# Prediction
if st.button("Predict"):
    result = model.predict(features)

    if result[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
