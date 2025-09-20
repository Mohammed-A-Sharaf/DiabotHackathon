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
        st.error(f"Error loading model: {e}")
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
    st.title("HealthGuard AI")
    page = st.radio("Navigation", ["Health Analysis", "AI Health Assistant"])

    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Total Patients", "342")
    st.metric("High Risk Patients", "27")
    st.metric("Avg HbA1c", "6.8%")

# -----------------------------
# Normalization Helper
# -----------------------------
NORMALIZATION_PARAMS = {
    "BMI": {"min": 12.0, "max": 98.0},
    "PhysHlth": {"min": 0.0, "max": 30.0},
    "MentHlth": {"min": 0.0, "max": 30.0}
}

def normalize_feature(value, feature_name):
    """Normalize feature using the same parameters as during training"""
    if feature_name in NORMALIZATION_PARAMS:
        min_val = NORMALIZATION_PARAMS[feature_name]["min"]
        max_val = NORMALIZATION_PARAMS[feature_name]["max"]
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    return value

def yes_no_to_binary(value):
    return 1 if value == "Yes" else 0

# -----------------------------
# Health Analysis Page
# -----------------------------
if page == "Health Analysis":
    st.markdown("## Patient Health Analysis & Risk Prediction")

    # Patient info
    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_name = st.text_input("Patient Name", "John Doe")
        age = st.number_input("Age", min_value=1, max_value=120, value=52)
        gender = st.radio("Gender", ["Male", "Female"])
    with col2:
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=175)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=82)
        bmi = round(weight / ((height / 100) ** 2), 1)
        st.write(f"**BMI:** {bmi}")
    with col3:
        last_checkup = st.date_input("Last Checkup")
        next_appointment = st.date_input("Next Appointment")
        status = st.selectbox("Health Status", ["Healthy", "Pre-Diabetic", "Diabetic"])

    st.markdown("---")

    st.subheader("Input Health Information")

    col1, col2, col3 = st.columns(3)

    # Medical History
    with col1:
        HighBP = st.radio("High Blood Pressure?", ["No", "Yes"])
        HighChol = st.radio("High Cholesterol?", ["No", "Yes"])
        CholCheck = st.radio("Cholesterol Check in last 5 years?", ["No", "Yes"])
        Stroke = st.radio("History of Stroke?", ["No", "Yes"])
        HeartDiseaseorAttack = st.radio("History of Heart Disease or Attack?", ["No", "Yes"])

    # Lifestyle
    with col2:
        Smoker = st.radio("Smoked 100+ cigarettes?", ["No", "Yes"])
        PhysActivity = st.radio("Physical Activity past 30 days?", ["No", "Yes"])
        Fruits = st.radio("Eat Fruits daily?", ["No", "Yes"])
        Veggies = st.radio("Eat Vegetables daily?", ["No", "Yes"])
        HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption?", ["No", "Yes"])
        GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    # Demographics
    with col3:
        Sex = 1 if gender == "Male" else 0
        
        # Age category with tooltip
        Age = st.slider(
            "Age category", 
            1, 13, 5,
            help="1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49, 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80+"
        )
        
        Education = st.slider("Education (1=Never attended, 6=College graduate)", 1, 6, 4)
        
        # Income with tooltip
        Income = st.slider(
            "Income", 
            1, 8, 4,
            help="1=<$10k, 2=$10k-15k, 3=$15k-20k, 4=$20k-25k, 5=$25k-35k, 6=$35k-50k, 7=$50k-75k, 8=$75k+"
        )
        
        NoDocbcCost = st.radio("Skipped doctor due to cost?", ["No", "Yes"])
        AnyHealthcare = st.radio("Healthcare Coverage?", ["No", "Yes"])
        DiffWalk = st.radio("Difficulty Walking?", ["No", "Yes"])

    # Health Metrics
    st.subheader("Health Metrics")
    col4, col5, col6 = st.columns(3)
    with col4:
        BMI = bmi
    with col5:
        PhysHlth = st.slider("Physical Health (days unwell past 30)", 0, 30, 5)
    with col6:
        MentHlth = st.slider("Mental Health (days unwell past 30)", 0, 30, 5)

    # Preprocess inputs
    features = [
        yes_no_to_binary(HighBP), 
        yes_no_to_binary(HighChol), 
        yes_no_to_binary(CholCheck), 
        normalize_feature(BMI, "BMI"), 
        yes_no_to_binary(Smoker), 
        yes_no_to_binary(Stroke), 
        yes_no_to_binary(HeartDiseaseorAttack), 
        yes_no_to_binary(PhysActivity), 
        yes_no_to_binary(Fruits), 
        yes_no_to_binary(Veggies), 
        yes_no_to_binary(HvyAlcoholConsump), 
        yes_no_to_binary(AnyHealthcare), 
        yes_no_to_binary(NoDocbcCost), 
        GenHlth, 
        normalize_feature(MentHlth, "MentHlth"), 
        normalize_feature(PhysHlth, "PhysHlth"), 
        yes_no_to_binary(DiffWalk), 
        Sex, 
        Age, 
        Education, 
        Income
    ]
    
    # Convert to tensor
    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Prediction
    if st.button("Predict Risk"):
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            risk = probabilities[0][1].item()

        st.success(f"Predicted Diabetes Risk: {risk:.2%}")

        # Risk interpretation
        if risk < 0.25:
            st.info("Low Risk – Maintain your healthy lifestyle.")
        elif risk < 0.6:
            st.warning("Moderate Risk – Consider lifestyle improvements.")
        else:
            st.error("High Risk – Please consult a healthcare professional.")

        # Probability breakdown
        st.progress(risk)
        st.write("**Probability Breakdown:**")
        st.write(f"- No Diabetes: {(1-risk):.2%}")
        st.write(f"- Diabetes: {risk:.2%}")

        # Additional insights
        if HighBP == "Yes":
            st.write("💡 **Note:** High blood pressure is a significant risk factor for diabetes.")
        if BMI >= 30:
            st.write("💡 **Note:** A BMI of 30 or higher increases diabetes risk.")
        if PhysActivity == "No":
            st.write("💡 **Note:** Regular physical activity can help reduce diabetes risk.")


# -----------------------------
# AI Health Assistant Page
# -----------------------------
elif page == "AI Health Assistant":
    st.markdown("## AI Health Assistant")
    
    st.info("This assistant provides personalized health recommendations based on your risk factors.")
    
    # Simple risk assessment
    st.subheader("Quick Health Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        family_history = st.radio("Family history of diabetes?", ["No", "Yes"])
        activity_level = st.radio("Physical activity level?", ["Sedentary", "Moderate", "Active"])
        diet_quality = st.radio("How would you rate your diet?", ["Poor", "Average", "Good"])
    
    with col2:
        sleep_hours = st.slider("Average hours of sleep per night", 3, 12, 7)
        stress_level = st.slider("Stress level (1=Low, 10=High)", 1, 10, 5)
        
    if st.button("Get Health Recommendations"):
        recommendations = []
        
        if activity_level == "Sedentary":
            recommendations.append("🏃‍♂️ **Increase physical activity**: Aim for at least 30 minutes of moderate exercise most days.")
        
        if diet_quality in ["Poor", "Average"]:
            recommendations.append("🥗 **Improve diet**: Focus on whole foods, fruits, vegetables, and limit processed foods.")
            
        if sleep_hours < 7:
            recommendations.append("😴 **Prioritize sleep**: Aim for 7-9 hours of quality sleep per night.")
            
        if stress_level > 7:
            recommendations.append("🧘‍♂️ **Manage stress**: Try meditation, deep breathing, or other relaxation techniques.")
            
        if family_history == "Yes":
            recommendations.append("👨‍👩‍👧‍👦 **Family history**: Be extra vigilant about regular check-ups due to your family history.")
            
        if recommendations:
            st.success("### Personalized Recommendations")
            for rec in recommendations:
                st.write(rec)
        else:
            st.info("You're doing great! Keep up your healthy habits.")
