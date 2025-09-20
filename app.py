import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# -----------------------------
# Model Definition
# -----------------------------
class DiabetesModel(nn.Module):
    def __init__(self, input_dim=21):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
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


# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    model = DiabetesModel(input_dim=21)
    try:
        model.load_state_dict(torch.load("Diabetes_model.pth", map_location="cpu"))
        model.eval()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
    return model

model = load_model()

# -----------------------------
# Page Config & Sidebar
# -----------------------------
st.set_page_config(
    page_title="HealthGuard AI - Diabetes Risk Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
with st.sidebar:
    st.title("HealthGuard AI 🏥")
    page = st.radio("Navigation", ["Health Analysis", "AI Health Assistant"])

    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Total Patients", "342")
    st.metric("High Risk Patients", "27")
    st.metric("Avg HbA1c", "6.8%")

# -----------------------------
# Normalization Helper
# -----------------------------
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0


# -----------------------------
# Health Analysis Page
# -----------------------------
if page == "Health Analysis":
    st.markdown("## 🩺 Patient Health Analysis & Risk Prediction")

    # Patient info (this should eventually be pulled from DB)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Patient:** John Doe")
        st.markdown("**Age:** 52 years")
        st.markdown("**Gender:** Male")
    with col2:
        st.markdown("**Height:** 175 cm")
        st.markdown("**Weight:** 82 kg")
        st.markdown("**BMI:** 26.8")
    with col3:
        st.markdown("**Last Checkup:** 15 days ago")
        st.markdown("**Next Appointment:** In 2 weeks")
        st.markdown("**Status:** Pre-Diabetic")

    st.markdown("---")

    st.subheader("📋 Input Health Information")

    col1, col2, col3 = st.columns(3)

    # Medical History
    with col1:
        HighBP = st.radio("High Blood Pressure?", [0, 1])
        HighChol = st.radio("High Cholesterol?", [0, 1])
        Stroke = st.radio("History of Stroke?", [0, 1])
        HeartDisease = st.radio("History of Heart Disease?", [0, 1])
        DiffWalk = st.radio("Difficulty Walking?", [0, 1])

    # Lifestyle
    with col2:
        Smoker = st.radio("Smoked 100+ cigarettes?", [0, 1])
        PhysActivity = st.radio("Physical Activity past 30 days?", [0, 1])
        Fruits = st.radio("Eat Fruits daily?", [0, 1])
        Veggies = st.radio("Eat Vegetables daily?", [0, 1])
        HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption?", [0, 1])
        GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    # Demographics
    with col3:
        Sex = st.radio("Sex (0=Female, 1=Male)", [0, 1])
        Age = st.slider("Age category (1=18-24, 13=80+)", 1, 13, 5)
        Education = st.slider("Education (1=Never attended, 6=College graduate)", 1, 6, 4)
        Income = st.slider("Income (1=<$10k, 8=$75k+)", 1, 8, 4)
        NoDocbcCost = st.radio("Skipped doctor due to cost?", [0, 1])
        AnyHealthcare = st.radio("Healthcare Coverage?", [0, 1])

    # BMI, Physical & Mental Health
    st.subheader("📊 Health Metrics")
    col4, col5, col6 = st.columns(3)
    with col4:
        BMI = st.slider("BMI (0–100)", 10, 50, 25)
    with col5:
        PhysHlth = st.slider("Physical Health (days unwell past 30)", 0, 30, 5)
    with col6:
        MentHlth = st.slider("Mental Health (days unwell past 30)", 0, 30, 5)

    # Preprocess inputs
    BMI = normalize(BMI, 10, 50)
    PhysHlth = normalize(PhysHlth, 0, 30)
    MentHlth = normalize(MentHlth, 0, 30)

    features = [
        HighBP, HighChol, BMI, Smoker, Stroke, HeartDisease, PhysActivity,
        Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
        GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
    ]
    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Prediction
    if st.button("🔮 Predict Risk"):
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            risk = probabilities[0][1].item()

        st.success(f"**Predicted Diabetes Risk: {risk:.2%}**")

        # Risk interpretation
        if risk < 0.25:
            st.info("🟢 Low Risk – Maintain your healthy lifestyle!")
        elif risk < 0.6:
            st.warning("🟠 Moderate Risk – Consider lifestyle improvements.")
        else:
            st.error("🔴 High Risk – Please consult a healthcare professional.")

        # Probability breakdown
        st.progress(risk)
        st.write("**Probability Breakdown:**")
        st.write(f"- No Diabetes: {(1-risk):.2%}")
        st.write(f"- Diabetes: {risk:.2%}")


# -----------------------------
# AI Health Assistant Page
# -----------------------------
elif page == "AI Health Assistant":
    st.markdown("## 🤖 AI Health Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask me anything about your health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Simple rule-based assistant (can be replaced with real AI backend)
        if "blood sugar" in prompt.lower():
            response = "Your blood sugar trends suggest careful monitoring. Try to maintain a balanced diet and regular exercise."
        elif "diet" in prompt.lower():
            response = "Include more fiber, lean proteins, and avoid excess sugar. Would you like a sample meal plan?"
        elif "exercise" in prompt.lower():
            response = "Daily 30-minute walks or moderate physical activity can significantly reduce diabetes risk."
        else:
            response = "I can provide insights about your blood sugar, diet, exercise, or medication. What would you like to focus on?"

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
