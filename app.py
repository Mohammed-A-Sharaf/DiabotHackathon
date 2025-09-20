import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import torch
import torch.nn as nn

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="HealthGuard AI - Diabetes Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; padding-bottom: 10px;}
    .section-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding: 10px 0;}
    .metric-label {font-weight: bold; color: #1f77b4;}
    .alert {padding: 10px; background-color: #ffcccc; border-radius: 5px; margin: 10px 0;}
    .good {padding: 10px; background-color: #ccffcc; border-radius: 5px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Define Model
# ------------------------------
class DiabetesModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=1):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load model
model = DiabetesModel()
try:
    model.load_state_dict(torch.load("Diabetes_model.pth", map_location="cpu"))
    model.eval()
except Exception as e:
    st.error("⚠️ Error loading model: " + str(e))

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.title("HealthGuard AI 🏥")
    st.markdown("### Patient Dashboard")

    patient_options = ["John Doe (ID: 12345)", "Jane Smith (ID: 67890)", "Robert Johnson (ID: 54321)"]
    selected_patient = st.selectbox("Select Patient", patient_options)

    selected_date = st.date_input("Select Date", datetime.now())

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Health Overview", "Detailed Analysis", "Appointment Scheduling", "AI Health Assistant"])

    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Total Patients", "342")
    st.metric("High Risk Patients", "27")
    st.metric("Avg HbA1c", "6.8%")

# ------------------------------
# Health Overview Page
# ------------------------------
if page == "Health Overview":
    st.markdown('<p class="main-header">Patient Health Dashboard</p>', unsafe_allow_html=True)

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
    st.markdown('<p class="section-header">Organ Health Status</p>', unsafe_allow_html=True)

    organs = ['Lungs', 'Stomach', 'Liver', 'Heart', 'Brain']
    health_values = [85, 72, 65, 78, 90]

    org_cols = st.columns(5)
    for i, col in enumerate(org_cols):
        with col:
            color = "green" if health_values[i] > 80 else "orange" if health_values[i] > 60 else "red"
            st.markdown(f"<h3 style='text-align: center;'>{organs[i]}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: {color};'>{health_values[i]}%</h2>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Blood Sugar Tracking</p>', unsafe_allow_html=True)

    dates = pd.date_range(start=selected_date - timedelta(days=30), end=selected_date)
    sugar_levels = [random.randint(100, 180) for _ in range(len(dates))]
    fig = px.line(x=dates, y=sugar_levels,
                  labels={'x': 'Date', 'y': 'Blood Sugar (mg/dL)'},
                  title="Blood Sugar Levels Over Time")
    fig.add_hrect(y0=70, y1=140, line_width=0, fillcolor="green", opacity=0.2)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Detailed Analysis + Model
# ------------------------------
elif page == "Detailed Analysis":
    st.markdown('<p class="main-header">Detailed Health Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">AI-Powered Diabetes Prediction</p>', unsafe_allow_html=True)

    # Input fields for model
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 0, 200, 120)
        bp = st.number_input("Blood Pressure", 0, 150, 70)
    with col2:
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 0, 120, 35)

    if st.button("Run Prediction"):
        try:
            X = torch.tensor([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], dtype=torch.float32)
            with torch.no_grad():
                prob = model(X).item()
            st.metric("Prediction Probability", f"{prob*100:.2f}%")
            if prob > 0.5:
                st.markdown('<div class="alert"><b>High Risk:</b> The model predicts a high chance of diabetes.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="good"><b>Low Risk:</b> The model predicts a low chance of diabetes.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error running prediction: {e}")

# ------------------------------
# Appointment Scheduling Page
# ------------------------------
elif page == "Appointment Scheduling":
    st.markdown('<p class="main-header">Appointment Scheduling</p>', unsafe_allow_html=True)
    st.write("📅 Schedule and view your medical appointments here.")

# ------------------------------
# AI Health Assistant Page
# ------------------------------
elif page == "AI Health Assistant":
    st.markdown('<p class="main-header">AI Health Assistant</p>', unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about your health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = "I'm here to help! Based on your records, please consult your doctor for personalized advice."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("**HealthGuard AI** | *Predictive Healthcare Analytics*")
