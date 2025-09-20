import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# -----------------------------
# Load Model
# -----------------------------
class DiabetesNet(nn.Module):
    def __init__(self, input_dim=21):
        super(DiabetesNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)   # extra layer
        self.out = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.out(x)
        return x

# Load trained model
model = DiabetesNet(input_dim=21)
model.load_state_dict(torch.load("Diabetes_model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Interactive Diabetes Risk Prediction")
st.markdown("Answer questions about your health, lifestyle, and demographics to see your diabetes risk.")

# -----------------------------
# Input Sections
# -----------------------------
st.sidebar.header("⚙️ Settings")
st.sidebar.info("Fill in the details in each section, then click **Predict Risk**.")

col1, col2, col3 = st.columns(3)

# Medical history
with col1:
    st.subheader("🏥 Medical History")
    HighBP = st.radio("High Blood Pressure?", [0, 1])
    HighChol = st.radio("High Cholesterol?", [0, 1])
    CholCheck = st.radio("Had cholesterol check in last 5 years?", [0, 1])  # ✅ added
    Stroke = st.radio("History of Stroke?", [0, 1])
    HeartDisease = st.radio("History of Heart Disease?", [0, 1])
    DiffWalk = st.radio("Difficulty Walking/Climbing Stairs?", [0, 1])

# Lifestyle
with col2:
    st.subheader("💡 Lifestyle")
    Smoker = st.radio("Smoked 100+ cigarettes in lifetime?", [0, 1])
    PhysActivity = st.radio("Physical Activity past 30 days?", [0, 1])
    Fruits = st.radio("Eat Fruits daily?", [0, 1])
    Veggies = st.radio("Eat Vegetables daily?", [0, 1])
    HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption?", [0, 1])
    GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

# Demographics & Access
with col3:
    st.subheader("🌍 Demographics & Access")
    Sex = st.radio("Sex (0=Female, 1=Male)", [0, 1])
    Age = st.slider("Age category (1=18-24, 13=80+)", 1, 13, 5)
    Education = st.slider("Education (1=Never attended, 6=College graduate)", 1, 6, 4)
    Income = st.slider("Income (1=<$10k, 8=$75k+)", 1, 8, 4)
    NoDocbcCost = st.radio("Skipped doctor due to cost?", [0, 1])
    AnyHealthcare = st.radio("Do you have Healthcare Coverage?", [0, 1])

# BMI, Physical Health, Mental Health
st.subheader("📊 Health Metrics")
col4, col5, col6 = st.columns(3)
with col4:
    BMI = st.slider("BMI (0–100)", 10, 50, 25)
with col5:
    PhysHlth = st.slider("Physical Health (days unwell in past 30)", 0, 30, 5)
with col6:
    MentHlth = st.slider("Mental Health (days unwell in past 30)", 0, 30, 5)

# -----------------------------
# Preprocessing
# -----------------------------
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

BMI = normalize(BMI, 10, 50)
PhysHlth = normalize(PhysHlth, 0, 30)
MentHlth = normalize(MentHlth, 0, 30)

# ✅ Now we have all 21 features
features = [
    HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDisease, PhysActivity,
    Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
    GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
]

X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Risk"):
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)
        risk = probabilities[0][1].item()

    st.success(f"**Predicted Diabetes Risk: {risk:.2%}**")

    if risk < 0.25:
        st.info("🟢 Low Risk – Maintain your healthy lifestyle!")
    elif risk < 0.6:
        st.warning("🟠 Moderate Risk – Consider lifestyle improvements.")
    else:
        st.error("🔴 High Risk – Please consult a healthcare professional.")

    st.progress(risk)
    st.write("**Probability Breakdown:**")
    st.write(f"- No Diabetes: {(1-risk):.2%}")
    st.write(f"- Diabetes: {risk:.2%}")
