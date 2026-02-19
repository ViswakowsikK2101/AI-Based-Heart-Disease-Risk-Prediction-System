
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

st.title("💓 AI-Based Heart Disease Risk Prediction System")
st.markdown("### Clinical Decision Support Tool")

st.write("Enter patient clinical details below to predict heart disease risk.")

# Load Dataset (Make sure heart.csv is present)
df = pd.read_csv("heart_cleveland_upload.csv")

# Feature Engineering (Same as training)
df['age_chol'] = df['age'] * df['chol']
df['bp_chol'] = df['trestbps'] * df['chol']
df['heart_stress'] = df['thalach'] - df['age']

X = df.drop("condition", axis=1)
y = df["condition"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

model.fit(X_scaled, y)

# UI Layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol Level", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
    restecg = st.slider("Resting ECG (0-2)", 0, 2, 1)

with col2:
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.slider("Slope of ST Segment (0-2)", 0, 2, 1)
    ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.slider("Thalassemia (0-3)", 0, 3, 1)

# Feature Engineering for input
age_chol = age * chol
bp_chol = trestbps * chol
heart_stress = thalach - age

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal,
                        age_chol, bp_chol, heart_stress]])

input_scaled = scaler.transform(input_data)

if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result:")

    if probability < 0.3:
        st.success(f"Low Risk ({probability*100:.2f}%)")
    elif probability < 0.6:
        st.warning(f"Moderate Risk ({probability*100:.2f}%)")
    else:
        st.error(f"High Risk ({probability*100:.2f}%)")

    st.write("Probability of Heart Disease:", f"{probability*100:.2f}%")
