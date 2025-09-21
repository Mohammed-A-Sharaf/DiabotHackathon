import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import boto3
import re
import matplotlib.pyplot as plt

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
        /* Sidebar button styling */
    .sidebar-button {
        display: block;
        width: 100%;
        padding: 12px 16px;
        margin: 8px 0;
        background-color: #1e40af;
        color: white;
        border: none;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sidebar-button:hover {
        background-color: #1e3a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .sidebar-button.active {
        background-color: #1e3a8a;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        transform: translateY(0);
    }
    
    .sidebar-section {
        margin: 20px 0;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .sidebar-title {
        color: white;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    .sidebar-link {
        color: #93c5fd;
        text-decoration: none;
        display: block;
        margin: 8px 0;
        transition: color 0.2s ease;
    }
    
    .sidebar-link:hover {
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e40af;
        color: white;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: white;
    }
    
    .css-1d391kg a {
        color: #93c5fd;
    }
    
    .css-1d391kg a:hover {
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1e40af;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1e3a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1e40af;
    }
    
    /* Card-like styling for sections */
    .block-container {
        padding: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e5e7eb;
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        color: #374151;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e40af;
        color: white;
    }
    
    .stTabs [data-baseweb="tab"] > div {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    
    .rtl-text {
        text-align: right;
        direction: rtl;
    }
    
    .stChatInput > div > div > input {
        text-align: left;
        direction: ltr;
    }
    
    .arabic-input .stChatInput > div > div > input {
        text-align: right;
        direction: rtl;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"] {
        padding: 1rem;
    }
    
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stSuccess {
        background-color: #dcfce7;
        color: #166534;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stError {
        background-color: #fee2e2;
        color: #991b1b;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #fef3c7;
        color: #92400e;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stProgress > div > div {
        background-color: #1e40af;
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    .stSlider > div > div {
        border-radius: 8px;
    }
    
    .health-good {
        color: #16a34a;
        font-weight: bold;
    }
    
    .health-warning {
        color: #ca8a04;
        font-weight: bold;
    }
    
    .health-danger {
        color: #dc2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
    page_title="DiaBot AI - Diabetes Risk Dashboard",
    page_icon="Logo Header/HeaderLogo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
with st.sidebar:
    st.image("DIaBot Logo/logo.png", width=200)
    st.markdown("---")
    
    # Initialize page in session state if not exists
    if 'page' not in st.session_state:
        st.session_state.page = "Health Analysis"
    
    # Navigation buttons - stacked vertically
    if st.button("Health Analysis", use_container_width=True, 
                type="primary" if st.session_state.page == "Health Analysis" else "secondary"):
        st.session_state.page = "Health Analysis"
        st.rerun()
    
    if st.button("AI Health Assistant", use_container_width=True,
                type="primary" if st.session_state.page == "AI Health Assistant" else "secondary"):
        st.session_state.page = "AI Health Assistant"
        st.rerun()
    
    if st.button("Health Education", use_container_width=True,
                type="primary" if st.session_state.page == "Health Education" else "secondary"):
        st.session_state.page = "Health Education"
        st.rerun()
    
    st.markdown("---")
    
    # Malaysian Resources section
    st.markdown('<div class="sidebar-title">Malaysian Resources</div>', unsafe_allow_html=True)
    st.markdown('<a href="https://www.moh.gov.my/" class="sidebar-link" target="_blank">Ministry of Health Malaysia</a>', unsafe_allow_html=True)
    st.markdown('<a href="http://www.nadi.org.my/" class="sidebar-link" target="_blank">National Diabetes Institute (NADI)</a>', unsafe_allow_html=True)
    st.markdown('<a href="http://www.diabetes.org.my/" class="sidebar-link" target="_blank">Malaysian Diabetes Association</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Emergency Contacts section
    st.markdown('<div class="sidebar-title">Emergency Contacts (Malaysia)</div>', unsafe_allow_html=True)
    st.markdown('**If you\'re experiencing a medical emergency, call 999 immediately.**')
    
    # Use HTML to prevent line breaks in phone numbers
    st.markdown("""
    <div style="margin-top: 10px;">
        <div>Health Advisory: <span style="white-space: nowrap;">03-8881 0200</span></div>
        <div>Poison Control: <span style="white-space: nowrap;">04-657 0099</span></div>
        <div>Mental Health: <span style="white-space: nowrap;">03-7956 8145</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Set page based on session state
page = st.session_state.page

# -----------------------------
# Normalization Helper
# -----------------------------
NORMALIZATION_PARAMS = {
    "BMI": {"min": 12.0, "max": 98.0},
    "PhysHlth": {"min": 0.0, "max": 30.0},
    "MentHlth": {"min": 0.0, "max": 30.0}
}

def normalize_feature(value, feature_name):
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
    1: "Less than RM 1,000",
    2: "RM 1,000 to RM 2,000",
    3: "RM 2,000 to RM 3,000",
    4: "RM 3,000 to RM 4,000",
    5: "RM 4,000 to RM 5,000",
    6: "RM 5,000 to RM 6,000",
    7: "RM 6,000 to RM 7,000",
    8: "RM 7,000 or more"
}

# Helper function for future predictions
def get_age_category(age):
    if age >= 80:
        return 13
    elif age >= 75:
        return 12
    elif age >= 70:
        return 11
    elif age >= 65:
        return 10
    elif age >= 60:
        return 9
    elif age >= 55:
        return 8
    elif age >= 50:
        return 7
    elif age >= 45:
        return 6
    elif age >= 40:
        return 5
    elif age >= 35:
        return 4
    elif age >= 30:
        return 3
    elif age >= 25:
        return 2
    else:
        return 1

# Function to predict future risk
def predict_future_risk(current_features, current_age, months=36):
    future_risks = []
    
    for month in range(0, months + 1, 6):  # Predict every 6 months
        future_features = current_features.copy()
        future_age = current_age + (month / 12)
        future_age_category = get_age_category(future_age)
        future_features[18] = future_age_category
        
        X_future = torch.tensor(future_features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(X_future)
            probabilities = torch.softmax(outputs, dim=1)
            risk = probabilities[0][1].item()
        
        future_risks.append(risk)
    
    return future_risks

# -----------------------------
# Health Analysis Page
# -----------------------------
if page == "Health Analysis":
    st.markdown("## Patient Health Analysis & Risk Prediction")

    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_name = st.text_input("Patient Name", "")
        age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter age")
        gender = st.radio("Gender", ["Male", "Female"], index=None)
    with col2:
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=None, placeholder="Enter height")
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=None, placeholder="Enter weight")
        if height and weight:
            bmi = round(weight / ((height / 100) ** 2), 1)
            st.write(f"**BMI:** {bmi}")
        else:
            bmi = None
            st.write("**BMI:** Please enter height and weight")
    with col3:
        last_checkup = st.date_input("Last Checkup", value=None)

    # Automatically calculate age category based on age
    if age is not None:
        age_category = get_age_category(age)
        age_category_display = age_categories.get(age_category, "Unknown")
        st.info(f"**Age Category:** {age_category} - {age_category_display}")
    else:
        age_category = None
        st.info("**Age Category:** Please enter age")

    st.markdown("---")
    st.subheader("Health Information")

    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Medical History", "Lifestyle", "Health Metrics"])

    with tab1:
        st.markdown("### Demographic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Sex = 1 if gender == "Male" else 0 if gender == "Female" else None
            
            Education = st.selectbox(
                "Education Level",
                options=[None] + list(range(1, 7)),
                format_func=lambda x: f"{x} - {['Never attended', 'Elementary', 'Some high school', 'High school graduate', 'Some college', 'College graduate'][x-1]}" if x is not None else "Select education level",
                index=0
            )
            
        with col2:
            Income = st.selectbox(
                "Income Category",
                options=[None] + list(income_categories.keys()),
                format_func=lambda x: f"{x} - {income_categories[x]}" if x is not None else "Select income category",
                index=0
            )
            
        with col3:
            AnyHealthcare = st.radio("Healthcare Coverage?", ["No", "Yes"], index=None)
            NoDocbcCost = st.radio("Skipped doctor due to cost?", ["No", "Yes"], index=None)

    with tab2:
        st.markdown("### Medical History")
        col1, col2 = st.columns(2)
        
        with col1:
            HighBP = st.radio("High Blood Pressure?", ["No", "Yes"], index=None)
            HighChol = st.radio("High Cholesterol?", ["No", "Yes"], index=None)
            CholCheck = st.radio("Cholesterol Check in last 5 years?", ["No", "Yes"], index=None)
            
        with col2:
            Stroke = st.radio("History of Stroke?", ["No", "Yes"], index=None)
            HeartDiseaseorAttack = st.radio("History of Heart Disease or Attack?", ["No", "Yes"], index=None)
            DiffWalk = st.radio("Difficulty Walking?", ["No", "Yes"], index=None)

    with tab3:
        st.markdown("### Lifestyle Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Smoker = st.radio("Smoked 100+ cigarettes?", ["No", "Yes"], index=None)
            HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption?", ["No", "Yes"], index=None)
            
        with col2:
            PhysActivity = st.radio("Physical Activity past 30 days?", ["No", "Yes"], index=None)
            Fruits = st.radio("Eat Fruits daily?", ["No", "Yes"], index=None)
            
        with col3:
            Veggies = st.radio("Eat Vegetables daily?", ["No", "Yes"], index=None)
            GenHlth = st.selectbox(
                "General Health (1=Excellent, 5=Poor)",
                options=[None] + list(range(1, 6)),
                format_func=lambda x: f"{x} - {['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][x-1]}" if x is not None else "Select general health",
                index=0
            )

    with tab4:
        st.markdown("### Health Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if bmi is not None:
                BMI = bmi
                st.info(f"**BMI:** {bmi} (calculated from height and weight)")
            else:
                BMI = None
                st.info("**BMI:** Please enter height and weight")
            
        with col2:
            PhysHlth = st.slider("Physical Health (days unwell past 30)", 0, 30, 0)
            
        with col3:
            MentHlth = st.slider("Mental Health (days unwell past 30)", 0, 30, 0)

    # Validation function
    def validate_inputs():
        required_fields = {
            "Age": age is not None,
            "Gender": gender is not None,
            "Height": height is not None,
            "Weight": weight is not None,
            "Education": Education is not None,
            "Income": Income is not None,
            "AnyHealthcare": AnyHealthcare is not None,
            "NoDocbcCost": NoDocbcCost is not None,
            "HighBP": HighBP is not None,
            "HighChol": HighChol is not None,
            "CholCheck": CholCheck is not None,
            "Stroke": Stroke is not None,
            "HeartDiseaseorAttack": HeartDiseaseorAttack is not None,
            "DiffWalk": DiffWalk is not None,
            "Smoker": Smoker is not None,
            "HvyAlcoholConsump": HvyAlcoholConsump is not None,
            "PhysActivity": PhysActivity is not None,
            "Fruits": Fruits is not None,
            "Veggies": Veggies is not None,
            "GenHlth": GenHlth is not None
        }
        
        missing_fields = [field for field, filled in required_fields.items() if not filled]
        
        if missing_fields:
            st.error(f"Please fill in all required fields: {', '.join(missing_fields)}")
            return False
        return True

    predict_btn = st.button("Predict Risk", type="primary", use_container_width=True)

    if predict_btn:
        if not validate_inputs():
            st.stop()
            
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
            age_category,  # Use the automatically calculated age category
            Education, 
            Income
        ]
        
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            risk = probabilities[0][1].item()

        st.session_state.health_data = {
            "patient_name": patient_name,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "risk": f"{risk:.2%}",
            "HighBP": HighBP,
            "HighChol": HighChol,
            "CholCheck": CholCheck,
            "Stroke": Stroke,
            "HeartDiseaseorAttack": HeartDiseaseorAttack,
            "PhysActivity": PhysActivity,
            "Fruits": Fruits,
            "Veggies": Veggies,
            "HvyAlcoholConsump": HvyAlcoholConsump,
            "GenHlth": GenHlth,
            "MentHlth": MentHlth,
            "PhysHlth": PhysHlth,
            "DiffWalk": DiffWalk,
            "Smoker": Smoker,
            "AnyHealthcare": AnyHealthcare,
            "NoDocbcCost": NoDocbcCost,
            "Education": Education,
            "Income": Income
        }
        
        st.success(f"Predicted Diabetes Risk: {risk:.2%}")

        if risk < 0.25:
            st.markdown('<div class="stSuccess">Low Risk – Maintain your healthy lifestyle.</div>', unsafe_allow_html=True)
        elif risk < 0.6:
            st.markdown('<div class="stWarning">Moderate Risk – Consider lifestyle improvements.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stError">High Risk – Please consult a healthcare professional.</div>', unsafe_allow_html=True)

        st.progress(risk)
        st.write("**Probability Breakdown:**")
        st.write(f"- No Diabetes: {(1-risk):.2%}")
        st.write(f"- Diabetes: {risk:.2%}")

        st.subheader("Personalized Health Insights")
        
        insights = []
        
        if BMI < 18.5:
            insights.append("Your BMI suggests you're underweight. Consider a balanced diet with adequate nutrition.")
        elif BMI >= 25 and BMI < 30:
            insights.append("Your BMI indicates overweight. Even a 5-7% weight loss can significantly reduce diabetes risk.")
        elif BMI >= 30:
            insights.append("Your BMI indicates obesity, a major diabetes risk factor. Focus on gradual weight loss through diet and exercise.")
        
        if HighBP == "Yes":
            insights.append("Managing your high blood pressure is crucial. Reduce sodium intake and monitor your levels regularly.")
        
        if HighChol == "Yes":
            insights.append("High cholesterol increases diabetes risk. Consider reducing saturated fats and increasing fiber intake.")
        
        if PhysActivity == "No":
            insights.append("Regular physical activity (150 mins/week) can improve insulin sensitivity. Start with brisk walking.")
        
        if Fruits == "No" or Veggies == "No":
            insights.append("Aim for 5 servings of fruits and vegetables daily. They're rich in fiber and antioxidants that protect against diabetes.")
        
        if MentHlth > 7:
            insights.append("Your mental health days are elevated. Stress management techniques may help reduce diabetes risk.")
        
        if Smoker == "Yes":
            insights.append("Smoking increases insulin resistance. Consider cessation programs to reduce your diabetes risk.")
        
        if insights:
            for i, insight in enumerate(insights, 1):
                st.write(f"{i}. {insight}")
        else:
            st.write("Your health profile shows no significant risk factors. Maintain your healthy habits!")
        
        st.subheader("Personalized Recommendations")
        
        if risk < 0.25:
            st.write("• Continue your current healthy lifestyle")
            st.write("• Schedule annual checkups to monitor your health")
            st.write("• Maintain a balanced diet and regular exercise routine")
        elif risk < 0.6:
            st.write("• Consider increasing physical activity to 30 minutes daily")
            st.write("• Focus on whole foods and reduce processed food intake")
            st.write("• Monitor your blood sugar levels periodically")
        else:
            st.write("• Consult with a healthcare provider for a comprehensive plan")
            st.write("• Consider working with a nutritionist for meal planning")
            st.write("• Regular blood glucose monitoring is recommended")
        
        st.warning("**Important Notice:** These insights and recommendations are generated based on the information provided and are not a substitute for professional medical advice. Please consult with your healthcare provider for personalized medical guidance.")
        
        # Add the statistical prediction for the next 36 months
        with st.spinner("Calculating future risk projections..."):
            future_risks = predict_future_risk(features, age, months=36)
            
            months = list(range(0, 37, 6))
            current_date = pd.Timestamp.now()
            month_labels = [(current_date + pd.DateOffset(months=i)).strftime("%b %Y") for i in months]
            
            risk_data = pd.DataFrame({
                "Month": months,
                "Month_Label": month_labels,
                "Risk": future_risks
            })
            
            st.subheader("36-Month Diabetes Risk Projection")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.line_chart(risk_data, x="Month_Label", y="Risk")
            
            with col2:
                st.write("**Risk by Period:**")
                for i, risk_val in enumerate(future_risks):
                    st.write(f"{month_labels[i]}: {risk_val:.2%}")
            
            st.subheader("Future Risk Analysis")
            
            risk_change = future_risks[-1] - future_risks[0]
            
            if risk_change > 0.1:
                st.error(f"**Warning:** Your diabetes risk is projected to increase significantly by {month_labels[-1]} (+{risk_change:.2%}). Consider making lifestyle changes now to reduce this risk.")
            elif risk_change > 0.05:
                st.warning(f"**Notice:** Your diabetes risk is projected to increase by {month_labels[-1]} (+{risk_change:.2%}). Small lifestyle changes now can help mitigate this increase.")
            elif abs(risk_change) <= 0.05:
                st.info(f"**Stable:** Your diabetes risk is projected to remain relatively stable through {month_labels[-1]} ({risk_change:+.2%}).")
            else:
                st.success(f"**Improving:** Your diabetes risk is projected to decrease by {month_labels[-1]} ({risk_change:+.2%}). Keep up your healthy habits!")
            
            st.subheader("Long-Term Recommendations")
            
            if future_risks[-1] >= 0.6:
                st.write("• Develop a comprehensive long-term health plan with your doctor")
                st.write("• Consider regular monitoring of blood glucose levels")
                st.write("• Focus on sustainable weight management strategies")
                st.write("• Explore stress reduction techniques for long-term health")
            elif future_risks[-1] >= 0.25:
                st.write("• Set gradual health improvement goals")
                st.write("• Consider annual health check-ups to monitor progress")
                st.write("• Focus on maintaining a balanced diet and regular exercise")
                st.write("• Monitor key health indicators like blood pressure and cholesterol")
            else:
                st.write("• Continue your current healthy lifestyle habits")
                st.write("• Stay vigilant with regular health screenings")
                st.write("• Share your healthy habits with friends and family")
                st.write("• Consider preventive health measures as you age")
            
            st.warning("""
            **Important Notice:** These future projections are estimates based on your current health profile and the assumption that most factors remain constant except for age. 
            Actual future risk may vary significantly based on lifestyle changes, medical interventions, and other factors. 
            Regular consultation with healthcare professionals is essential for accurate health assessment and planning.
            """)
# -----------------------------
# AI Health Assistant Page with AWS Bedrock Chatbot
# -----------------------------
elif page == "AI Health Assistant":
    st.markdown("## AI Health Assistant")
    
    @st.cache_resource
    def get_bedrock_client():
        try:
            aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
            aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
            region = st.secrets["AWS_DEFAULT_REGION"]
            
            client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
            return client
        except Exception as e:
            st.error(f"Error initializing AWS Bedrock client: {e}")
            st.error("Please make sure your AWS credentials are correctly set in Streamlit secrets.")
            return None
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI health assistant specializing in diabetes care. Have you completed your health analysis yet? I can provide better advice if you share your health information with me.", "language": "English"}
        ]
    
    if "language" not in st.session_state:
        st.session_state.language = "English"
    
    if "show_quick_actions" not in st.session_state:
        st.session_state.show_quick_actions = True
    
    if "quick_action_triggered" not in st.session_state:
        st.session_state.quick_action_triggered = False
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.language = st.selectbox(
            "Select Chat Language",
            ["English", "Malay", "Chinese", "Tamil", "Arabic"],
            index=0
        )
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm your AI health assistant specializing in diabetes care. Have you completed your health analysis yet? I can provide better advice if you share your health information with me.", "language": "English"}
            ]
            st.session_state.show_quick_actions = True
            st.session_state.quick_action_triggered = False
            st.rerun()
    
    # Apply appropriate text direction based on language
    if st.session_state.language == "Arabic":
        st.markdown('<div class="arabic-input">', unsafe_allow_html=True)
    elif st.session_state.language in ["Chinese", "Japanese", "Korean"]:
        st.markdown('<div class="cjk-text">', unsafe_allow_html=True)
    
    health_data_exists = "health_data" in st.session_state
    
    if health_data_exists:
        st.info("### Your Health Summary")
        health_data = st.session_state.health_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Age:** {health_data.get('age', 'Not provided')}")
            st.write(f"**Gender:** {health_data.get('gender', 'Not provided')}")
            st.write(f"**BMI:** {health_data.get('bmi', 'Not provided')}")
            st.write(f"**Diabetes Risk:** {health_data.get('risk', 'Not calculated')}")
        
        with col2:
            st.write(f"**Blood Pressure:** {'High' if health_data.get('HighBP') == 'Yes' else 'Normal'}")
            st.write(f"**Cholesterol:** {'High' if health_data.get('HighChol') == 'Yes' else 'Normal'}")
            st.write(f"**Activity Level:** {'Active' if health_data.get('PhysActivity') == 'Yes' else 'Inactive'}")
            st.write(f"**General Health:** {health_data.get('GenHlth', 'Not provided')}/5")
    
    if st.session_state.show_quick_actions and len(st.session_state.messages) == 1:
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Get Diet Recommendations", help="Get personalized diet suggestions based on your health profile"):
                prompt = "Provide specific dietary recommendations for diabetes prevention"
                if health_data_exists:
                    prompt += f" for a {st.session_state.health_data.get('age')} year old {st.session_state.health_data.get('gender')} with a BMI of {st.session_state.health_data.get('bmi')}"
                st.session_state.messages.append({"role": "user", "content": prompt, "language": st.session_state.language})
                st.session_state.show_quick_actions = False
                st.session_state.quick_action_triggered = True
                st.rerun()
        
        with col2:
            if st.button("Exercise Plan", help="Get a personalized exercise plan"):
                prompt = "Suggest an appropriate exercise routine"
                if health_data_exists:
                    activity_level = "active" if st.session_state.health_data.get('PhysActivity') == 'Yes' else "sedentary"
                    prompt += f" for someone who is currently {activity_level}"
                st.session_state.messages.append({"role": "user", "content": prompt, "language": st.session_state.language})
                st.session_state.show_quick_actions = False
                st.session_state.quick_action_triggered = True
                st.rerun()
        
        with col3:
            if st.button("Risk Explanation", help="Understand your diabetes risk factors"):
                if health_data_exists:
                    prompt = f"Explain my diabetes risk of {st.session_state.health_data.get('risk')} and what factors contribute to it"
                else:
                    prompt = "What are the main risk factors for diabetes?"
                st.session_state.messages.append({"role": "user", "content": prompt, "language": st.session_state.language})
                st.session_state.show_quick_actions = False
                st.session_state.quick_action_triggered = True
                st.rerun()
    
    st.markdown("---")
    st.markdown("### Chat with Health Assistant")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("language") == "Arabic":
                st.markdown(f'<div class="rtl-text">{message["content"]}</div>', unsafe_allow_html=True)
            elif message.get("language") in ["Chinese", "Japanese", "Korean"]:
                st.markdown(f'<div class="cjk-text">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
    
    # Close the language-specific divs
    if st.session_state.language in ["Arabic", "Chinese", "Japanese", "Korean"]:
        st.markdown('</div>', unsafe_allow_html=True)
    
    def invoke_llama(prompt, max_tokens=800, temperature=0.5):
        try:
            bedrock_client = get_bedrock_client()
            if bedrock_client is None:
                return "Error connecting to AI service. Please try again later."
            
            language_instruction = f"Please respond in {st.session_state.language}." if st.session_state.language != "English" else ""
            
            formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
{language_instruction}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            
            body = json.dumps({
                "prompt": formatted_prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            })
            
            response = bedrock_client.invoke_model(
                modelId='meta.llama3-70b-instruct-v1:0',
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return response_body.get('generation', 'No response generated')
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_user_input(prompt):
        # Only add user message if it's not already the last message
        if len(st.session_state.messages) == 0 or st.session_state.messages[-1]["content"] != prompt:
            st.session_state.messages.append({"role": "user", "content": prompt, "language": st.session_state.language})
        
        with st.chat_message("user"):
            if st.session_state.language == "Arabic":
                st.markdown(f'<div class="rtl-text">{prompt}</div>', unsafe_allow_html=True)
            elif st.session_state.language in ["Chinese", "Japanese", "Korean"]:
                st.markdown(f'<div class="cjk-text">{prompt}</div>', unsafe_allow_html=True)
            else:
                st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                health_context = ""
                
                if health_data_exists:
                    health_data = st.session_state.health_data
                    health_context = f"""
Patient Health Context:
- Age: {health_data.get('age', 'Not provided')}
- Gender: {health_data.get('gender', 'Not provided')}
- BMI: {health_data.get('bmi', 'Not provided')}
- Diabetes Risk: {health_data.get('risk', 'Not calculated')}
- Blood Pressure: {'High' if health_data.get('HighBP') == 'Yes' else 'Normal'}
- Cholesterol: {'High' if health_data.get('HighChol') == 'Yes' else 'Normal'}
- Physical Activity: {'Active' if health_data.get('PhysActivity') == 'Yes' else 'Inactive'}
- Diet: Fruits: {'Yes' if health_data.get('Fruits') == 'Yes' else 'No'}, Vegetables: {'Yes' if health_data.get('Veggies') == 'Yes' else 'No'}
- General Health: {health_data.get('GenHlth', 'Not provided')}/5
- Smoking: {'Yes' if health_data.get('Smoker') == 'Yes' else 'No'}
- Alcohol: {'Heavy' if health_data.get('HvyAlcoholConsump') == 'Yes' else 'Moderate/None'}

"""
                
                language_instruction = ""
                if st.session_state.language != "English":
                    language_instruction = f"""
IMPORTANT LANGUAGE INSTRUCTION: 
- You MUST respond exclusively in {st.session_state.language}. 
- Do NOT include any words, phrases, or sentences in any other language.
- If you cannot respond fully in {st.session_state.language}, say so and ask the user to rephrase in English.
- This is critical for user understanding and safety.
"""
                
                full_prompt = f"""
You are a friendly and knowledgeable health assistant specializing in diabetes prevention and management.
Provide helpful, evidence-based advice about nutrition, exercise, and lifestyle changes.
Always remind users to consult healthcare professionals for medical advice.

{language_instruction}

{health_context}
Current conversation context: {st.session_state.messages[-3:] if len(st.session_state.messages) > 3 else 'New conversation'}

User question: {prompt}

Please provide a helpful, concise response focused on diabetes prevention and management.
"""
                
                if st.session_state.language != "English":
                    full_prompt += f"\n\nRemember: Respond ONLY in {st.session_state.language}."
                
                full_response = invoke_llama(full_prompt)
                
                # Handle incorrect language responses
                if st.session_state.language != "English":
                    if st.session_state.language in ["Chinese", "Japanese", "Korean"]:
                        latin_chars = sum(1 for c in full_response if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
                        total_chars = max(1, len(full_response))
                        if latin_chars / total_chars > 0.5:
                            retry_prompt = f"""
The previous response was not in {st.session_state.language}. Please provide a response in {st.session_state.language} ONLY.

Original question: {prompt}
"""
                            full_response = invoke_llama(retry_prompt)
                    else:
                        english_words = r'\b(if|the|and|or|but|is|are|was|were|to|for|of|in|on|at|by|with|about|against|between|into|through|during|before|after|above|below|from|up|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|can|will|just|don|should|now)\b'
                        english_matches = re.findall(english_words, full_response, re.IGNORECASE)
                        if english_matches and len(english_matches) > 3:
                            retry_prompt = f"""
The previous response contained mixed languages. Please provide a response in {st.session_state.language} ONLY.

Original question: {prompt}
"""
                            full_response = invoke_llama(retry_prompt)
                
                if st.session_state.language == "Arabic":
                    st.markdown(f'<div class="rtl-text">{full_response}</div>', unsafe_allow_html=True)
                elif st.session_state.language in ["Chinese", "Japanese", "Korean"]:
                    st.markdown(f'<div class="cjk-text">{full_response}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(full_response)
        
        # Only add assistant message if it's not already the last message
        if len(st.session_state.messages) == 0 or st.session_state.messages[-1]["content"] != full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response, "language": st.session_state.language})
        st.session_state.show_quick_actions = False
        st.session_state.quick_action_triggered = False
    
    # ✅ FIX: reset quick_action_triggered BEFORE processing
    if st.session_state.quick_action_triggered and len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_message = st.session_state.messages[-1]["content"]
        st.session_state.quick_action_triggered = False  # reset here
        process_user_input(user_message)
        st.rerun()
    
    chat_placeholder = st.empty()
    with chat_placeholder:
        prompt = st.chat_input("Ask about diabetes prevention, nutrition, or exercise...")
    
    if prompt:
        process_user_input(prompt)
        st.rerun()


# -----------------------------
# Health Education Page
# -----------------------------
elif page == "Health Education":
    st.markdown("# Health Education")
    st.markdown("Learn about diabetes prevention and management in Malaysia")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Diabetes Basics", "Malaysian Context", "Nutrition Guide", "Exercise & Lifestyle"])
    
    with tab1:
        st.markdown("## Understanding Diabetes")
        st.markdown("""
        ### What is Diabetes?
        Diabetes is a chronic condition that occurs when the pancreas doesn't produce enough insulin or when the body cannot effectively use the insulin it produces.
        
        ### Types of Diabetes
        - **Type 1 Diabetes**: Usually diagnosed in children and young adults
        - **Type 2 Diabetes**: Most common form, often related to lifestyle factors
        - **Gestational Diabetes**: Occurs during pregnancy
        
        ### Common Symptoms
        - Frequent urination
        - Excessive thirst
        - Extreme hunger
        - Unexplained weight loss
        - Fatigue
        - Blurred vision
        """)
    
    with tab2:
        st.markdown("## Diabetes in Malaysia")
        st.markdown("""
        ### Statistics
        - Malaysia has the highest rate of diabetes in Western Pacific
        - Approximately 3.9 million Malaysians aged 18+ have diabetes
        - Many cases remain undiagnosed
        
        ### Risk Factors for Malaysians
        - Genetic predisposition
        - Traditional diets high in carbohydrates and sugar
        - Sedentary lifestyles
        - Urbanization and changing food habits
        
        ### Government Initiatives
        - National Strategic Plan for Non-Communicable Diseases
        - MySejahtera health screening initiatives
        - Subsidized healthcare for diabetes management
        """)
    
    with tab3:
        st.markdown("## Malaysian Nutrition Guide")
        st.markdown("""
        ### Healthy Local Food Choices
        - **Nasi**: Choose brown rice over white rice
        - **Protein**: Opt for grilled fish or chicken instead of fried
        - **Vegetables**: Increase intake of ulam and local greens
        - **Fruits**: Enjoy local fruits like papaya, guava, and watermelon
        
        ### Foods to Limit
        - Sweet drinks like teh tarik and sirap
        - High-sugar kuih and desserts
        - Fried foods and high-fat dishes
        - Processed foods and snacks
        
        ### Portion Control Tips
        - Use the "suku-suku separuh" method: 1/4 protein, 1/4 carbs, 1/2 vegetables
        - Choose smaller portions of rice
        - Limit sugary beverages
        """)
    
    with tab4:
        st.markdown("## Exercise & Lifestyle")
        st.markdown("""
        ### Recommended Physical Activity
        - At least 150 minutes of moderate exercise per week
        - Brisk walking, cycling, or swimming
        - Traditional activities like silat or tai chi
        
        ### Incorporating Activity into Daily Life
        - Take the stairs instead of elevators
        - Walk during lunch breaks
        - Park farther from destinations
        - Join community exercise groups
        
        ### Stress Management
        - Practice mindfulness and meditation
        - Get adequate sleep (7-8 hours per night)
        - Maintain social connections
        - Seek professional help if needed
        """)
    
    st.markdown("---")
    st.markdown("## Additional Resources")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Malaysian Health Organizations")
        st.markdown("- [Ministry of Health Malaysia](https://www.moh.gov.my/)")
        st.markdown("- [National Diabetes Institute](http://www.nadi.org.my/)")
        st.markdown("- [Malaysian Diabetes Association](http://www.diabetes.org.my/)")
    
    with col2:
        st.markdown("### Educational Materials")
        st.markdown("- [Diabetes Malaysia Handbook](http://www.diabetes.org.my/article.php?aid=141)")
        st.markdown("- [Healthy Eating Guide](https://www.moh.gov.my/index.php/pages.view/227)")
        st.markdown("- [Exercise Recommendations](https://www.moh.gov.my/index.php/pages.view/229)")
