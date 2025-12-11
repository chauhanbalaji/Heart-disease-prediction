import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Load trained model & scaler ---
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Feature columns (must match training data order) ---
feature_cols = [
    'age','sex','chest_pain_type','resting_blood_pressure','cholesterol',
    'fasting_blood_sugar','resting_ecg','max_heart_rate','exercise_induced_angina',
    'st_depression','st_slope','num_major_vessels','thalassemia'
]

# --- Function to make predictions ---
def predict_heart_disease(input_data):
    # Convert input array to DataFrame to avoid warnings
    input_df = pd.DataFrame([input_data], columns=feature_cols)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of positive class
    return prediction, probability

# --- Streamlit UI ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter your health details below to check your risk of heart disease.")

# --- Input Form ---
with st.form("heart_form"):
    st.subheader("👤 Personal Information")
    age = st.number_input("Age (years)", 1, 120, 30)
    sex = st.radio("Sex", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0

    st.subheader("💓 Medical Information")
    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    chest_pain_type = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain_type)

    resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
    cholesterol = st.number_input("Cholesterol Level (mg/dl)", 100, 600, 200)
    fasting_blood_sugar = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    fasting_blood_sugar = 1 if fasting_blood_sugar == "Yes" else 0

    resting_ecg = st.selectbox("Resting ECG Result", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    resting_ecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(resting_ecg)

    max_heart_rate = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exercise_induced_angina = st.radio("Exercise Induced Angina?", ["No", "Yes"])
    exercise_induced_angina = 1 if exercise_induced_angina == "Yes" else 0

    st_depression = st.number_input("ST Depression (exercise vs rest)", 0.0, 10.0, 1.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    num_major_vessels = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    thalassemia = ["Normal", "Fixed Defect", "Reversible Defect"].index(thalassemia) + 1

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction ---
if submitted:
    input_data = [
        age, sex, chest_pain_type, resting_blood_pressure, cholesterol,
        fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_induced_angina,
        st_depression, st_slope, num_major_vessels, thalassemia
    ]

    prediction, probability = predict_heart_disease(input_data)

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk: Heart Disease Detected\nProbability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low Risk: No Heart Disease\nConfidence: {(1-probability)*100:.2f}%")
