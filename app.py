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

# Age category mapping
age_categories = {
    1: "18-24 years",
    2: "25-29 years",
    3: "30-34 years",
    4: "35-39 years",
    5: "40-44 years",
    6: "45-49 years",
    7: "50-54 years",
    8: "55-59 years",
    9: "60-64 years",
    10: "65-69 years",
    11: "70-74 years",
    12: "75-79 years",
    13: "80+ years"
}

# Income category mapping
income_categories = {
    1: "Less than $10,000",
    2: "$10,000 to $15,000",
    3: "$15,000 to $20,000",
    4: "$20,000 to $25,000",
    5: "$25,000 to $35,000",
    6: "$35,000 to $50,000",
    7: "$50,000 to $75,000",
    8: "$75,000 or more"
}

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

    st.subheader("Health Information")

    # Use tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Medical History", "Lifestyle", "Health Metrics"])

    with tab1:
        st.markdown("### Demographic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Sex = 1 if gender == "Male" else 0
            Age = st.selectbox(
                "Age Category",
                options=list(age_categories.keys()),
                format_func=lambda x: f"{x} - {age_categories[x]}",
                index=4  # Default to 40-44 years
            )
            
        with col2:
            Education = st.slider("Education Level (1=Never attended, 6=College graduate)", 1, 6, 4)
            Income = st.selectbox(
                "Income Category",
                options=list(income_categories.keys()),
                format_func=lambda x: f"{x} - {income_categories[x]}",
                index=3  # Default to $20,000 to $25,000
            )
            
        with col3:
            AnyHealthcare = st.radio("Healthcare Coverage?", ["No", "Yes"])
            NoDocbcCost = st.radio("Skipped doctor due to cost?", ["No", "Yes"])

    with tab2:
        st.markdown("### Medical History")
        col1, col2 = st.columns(2)
        
        with col1:
            HighBP = st.radio("High Blood Pressure?", ["No", "Yes"])
            HighChol = st.radio("High Cholesterol?", ["No", "Yes"])
            CholCheck = st.radio("Cholesterol Check in last 5 years?", ["No", "Yes"])
            
        with col2:
            Stroke = st.radio("History of Stroke?", ["No", "Yes"])
            HeartDiseaseorAttack = st.radio("History of Heart Disease or Attack?", ["No", "Yes"])
            DiffWalk = st.radio("Difficulty Walking?", ["No", "Yes"])

    with tab3:
        st.markdown("### Lifestyle Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Smoker = st.radio("Smoked 100+ cigarettes?", ["No", "Yes"])
            HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption?", ["No", "Yes"])
            
        with col2:
            PhysActivity = st.radio("Physical Activity past 30 days?", ["No", "Yes"])
            Fruits = st.radio("Eat Fruits daily?", ["No", "Yes"])
            
        with col3:
            Veggies = st.radio("Eat Vegetables daily?", ["No", "Yes"])
            GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    with tab4:
        st.markdown("### Health Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            BMI = bmi
            st.info(f"**BMI:** {bmi} (calculated from height and weight)")
            
        with col2:
            PhysHlth = st.slider("Physical Health (days unwell past 30)", 0, 30, 5)
            
        with col3:
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
    if st.button("Predict Risk", type="primary", use_container_width=True):
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
        st.subheader("Personalized Insights")
        if HighBP == "Yes":
            st.write("💡 **Blood Pressure:** High blood pressure is a significant risk factor for diabetes.")
        if BMI >= 30:
            st.write("💡 **Weight Management:** A BMI of 30 or higher increases diabetes risk.")
        if PhysActivity == "No":
            st.write("💡 **Physical Activity:** Regular physical activity can help reduce diabetes risk.")
        if HighChol == "Yes":
            st.write("💡 **Cholesterol:** High cholesterol levels can contribute to diabetes risk.")
        if Smoker == "Yes":
            st.write("💡 **Smoking:** Smoking increases the risk of developing diabetes.")
        if Fruits == "No" or Veggies == "No":
            st.write("💡 **Nutrition:** A diet rich in fruits and vegetables can help prevent diabetes.")


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
        
    if st.button("Get Health Recommendations", type="primary"):
        recommendations = []
        
        if activity_level == "Sedentary":
            recommendations.append("🏃‍♂️ **Increase physical activity:** Aim for at least 30 minutes of moderate exercise most days.")
        
        if diet_quality in ["Poor", "Average"]:
            recommendations.append("🥗 **Improve diet:** Focus on whole foods, fruits, vegetables, and limit processed foods.")
            
        if sleep_hours < 7:
            recommendations.append("😴 **Prioritize sleep:** Aim for 7-9 hours of quality sleep per night.")
            
        if stress_level > 7:
            recommendations.append("🧘‍♂️ **Manage stress:** Try meditation, deep breathing, or other relaxation techniques.")
            
        if family_history == "Yes":
            recommendations.append("👨‍👩‍👧‍👦 **Family history:** Be extra vigilant about regular check-ups due to your family history.")
            
        if recommendations:
            st.success("### Personalized Recommendations")
            for rec in recommendations:
                st.write(rec)
        else:
            st.info("You're doing great! Keep up your healthy habits.")
