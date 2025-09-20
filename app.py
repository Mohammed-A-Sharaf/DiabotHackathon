import streamlit as st
import torch
import numpy as np
from model import DiabetesNet  # <-- make sure model.py has DiabetesNet defined

# --- Load model ---
input_dim = 21
model = DiabetesNet(input_dim)
model.load_state_dict(torch.load("Diabetes_model.pth", map_location=torch.device("cpu")))
model.eval()

# --- Streamlit UI ---
st.set_page_config(page_title="🩺 Diabetes Risk Predictor", layout="wide")

st.title("🩺 Interactive Diabetes Risk Prediction")
st.markdown("Answer questions about your health, lifestyle, and demographics to see your diabetes risk.")

# --- Example inputs (just a few for now) ---
age = st.slider("Age (Years)", 18, 80, 30)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
smoker = st.radio("Smoked 100+ cigarettes in lifetime?", [0, 1])
phys_activity = st.radio("Physical Activity in past 30 days?", [0, 1])

# --- Prediction button ---
if st.button("Predict Diabetes Risk"):
    x = np.zeros((1, 21))  # placeholder for all 21 features
    x[0, 0] = age / 80     # normalized like preprocessing
    x[0, 1] = bmi / 50
    x[0, 2] = smoker
    x[0, 3] = phys_activity
    # TODO: add the other 17 features

    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    st.success(f"✅ Risk of diabetes: {probs[1]*100:.2f}%")
