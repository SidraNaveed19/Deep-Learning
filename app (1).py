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
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol Level", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible)", [1, 2, 3])

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
