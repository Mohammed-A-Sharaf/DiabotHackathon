import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# -----------------------------
# Model Definition
# -----------------------------
class DiabetesNet(nn.Module):
    def __init__(self, input_dim=21):
        super(DiabetesNet, self).__init__()
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

# Load Model Safely
@st.cache_resource
def load_model():
    try:
        model = DiabetesNet(input_dim=21)
        model.load_state_dict(torch.load("Diabetes_model.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        transform: scale(1.02);
    }
    .stCard {
        background-color: white;
        border-radius: 16px;
        padding: 1.5em;
        margin: 0.8em 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #1f2937;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Diabetes Risk Prediction Dashboard")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["📊 Detailed Analysis", "📁 Patient Data"])

# -----------------------------
# Tab 1: Detailed Analysis
# -----------------------------
with tabs[0]:
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.subheader("📊 Detailed Risk Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏥 Medical History")
        HighBP = st.radio("High Blood Pressure?", [0, 1])
        HighChol = st.radio("High Cholesterol?", [0, 1])
        Stroke = st.radio("History of Stroke?", [0, 1])
        HeartDisease = st.radio("Heart Disease?", [0, 1])
        DiffWalk = st.radio("Difficulty Walking?", [0, 1])

    with col2:
        st.subheader("💡 Lifestyle")
        Smoker = st.radio("Smoked 100+ cigarettes?", [0, 1])
        PhysActivity = st.radio("Physical Activity?", [0, 1])
        Fruits = st.radio("Eat Fruits daily?", [0, 1])
        Veggies = st.radio("Eat Vegetables daily?", [0, 1])
        HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption?", [0, 1])
        GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    with col3:
        st.subheader("🌍 Demographics & Access")
        Sex = st.radio("Sex (0=Female, 1=Male)", [0, 1])
        Age = st.slider("Age category (1=18-24, 13=80+)", 1, 13, 5)
        Education = st.slider("Education (1=None, 6=Graduate)", 1, 6, 4)
        Income = st.slider("Income (1=<$10k, 8=$75k+)", 1, 8, 4)
        NoDocbcCost = st.radio("Skipped doctor due to cost?", [0, 1])
        AnyHealthcare = st.radio("Have Healthcare Coverage?", [0, 1])

    st.subheader("📊 Health Metrics")
    col4, col5, col6 = st.columns(3)
    with col4:
        BMI = st.slider("BMI (0–100)", 10, 50, 25)
    with col5:
        PhysHlth = st.slider("Physical Health (days unwell)", 0, 30, 5)
    with col6:
        MentHlth = st.slider("Mental Health (days unwell)", 0, 30, 5)

    # Normalize
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

    BMI = normalize(BMI, 10, 50)
    PhysHlth = normalize(PhysHlth, 0, 30)
    MentHlth = normalize(MentHlth, 0, 30)

    features = [
        HighBP, HighChol, BMI, Smoker, Stroke, HeartDisease, PhysActivity,
        Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
        GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
    ]

    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    if st.button("🔮 Predict Risk"):
        if model is not None:
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
        else:
            st.error("⚠️ Model not loaded correctly.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Tab 2: Patient Data
# -----------------------------
with tabs[1]:
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.subheader("📁 Patient Data from Database")

    # Example data (replace with DB integration later)
    data = {
        "Patient ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [45, 52, 37],
        "BMI": [24.5, 29.1, 31.2],
        "Risk": ["Low", "High", "Moderate"]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
