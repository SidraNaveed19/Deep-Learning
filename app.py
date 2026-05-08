import streamlit as st
import pickle
import numpy as np

# Load model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Diabetes Prediction App")

st.write("Enter patient details below:")

# Inputs
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# Prepare input
features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

# Scale input
features = scaler.transform(features)

# Prediction
if st.button("Predict"):
    result = model.predict(features)

    if result[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
