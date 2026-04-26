import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

# Load trained model
model = joblib.load("Heart_disease_model.pkl")

# Set page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# App Title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter the following medical parameters to predict the presence of heart disease.")
with st.expander("📘 Feature Descriptions (Click to Expand)"):
    st.markdown("""
- **age** – Age in years  
- **sex** – (1 = male; 0 = female)  
- **cp** – Chest pain type  
    - 0: Typical angina – chest pain due to decreased blood supply  
    - 1: Atypical angina – chest pain not related to heart  
    - 2: Non-anginal pain – often esophageal spasms  
    - 3: Asymptomatic – no visible symptoms  
- **trestbps** – Resting blood pressure (mm Hg). Anything above 130–140 is concerning  
- **chol** – Serum cholesterol in mg/dl  
    - Serum = LDL + HDL + 0.2 × triglycerides  
    - >200 is a concern  
- **fbs** – Fasting blood sugar >120 mg/dl (1 = true; 0 = false)  
    - >126 mg/dl may indicate diabetes  
- **restecg** – Resting electrocardiographic results  
    - 0: Normal  
    - 1: ST-T wave abnormality (non-normal heartbeat)  
    - 2: Left ventricular hypertrophy (enlarged pumping chamber)  
- **thalach** – Max heart rate achieved  
- **exang** – Exercise-induced angina (1 = yes; 0 = no)  
- **oldpeak** – ST depression during exercise (indicates heart stress)  
- **slope** – Slope of the ST segment during peak exercise  
    - 0: Upsloping (uncommon)  
    - 1: Flat (typical healthy heart)  
    - 2: Downsloping (signs of heart disease)  
- **ca** – No. of major vessels (0–3) visible via fluoroscopy  
- **thal** – Thallium stress test result  
    - 1, 3: Normal  
    - 6: Fixed defect (previous issue, now okay)  
    - 7: Reversible defect (issue under stress/exercise)  
- **target** – Disease presence (1 = Yes, 0 = No)  
    """)


# Sidebar Info
with st.sidebar:
    st.header("🩺 About")
    st.write("""
        This app uses a machine learning model to predict the **likelihood of heart disease** 
        based on patient medical attributes.
    """)
    st.write("Developed by **Animesh**")

# Input fields
def user_input():
    age = st.slider("Age", 20, 90, 50)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Serum Cholestoral in mg/dl (chol)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2])  # or adjust if encoded differently

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame([data])

# Predict
input_df = user_input()

if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({round(pred_proba * 100, 2)}% confidence)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({round(pred_proba * 100, 2)}% confidence)")

    st.markdown("----")
    st.subheader("🧾 Input Summary")
    st.dataframe(input_df)

