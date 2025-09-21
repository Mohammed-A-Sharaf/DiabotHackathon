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
        padding: 12px 20px;  /* Increased padding for better spacing */
        color: #374151;
        display: flex;
        align-items: center;  /* Vertically center text */
        justify-content: center;  /* Horizontally center text */
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e40af;
        color: white;
    }
    
    /* Ensure tab text is properly aligned */
    .stTabs [data-baseweb="tab"] > div {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    
    /* Right-to-left alignment for Arabic messages */
    .rtl-text {
        text-align: right;
        direction: rtl;
    }
    
    /* Adjust chat input for RTL when Arabic is selected */
    .stChatInput > div > div > input {
        text-align: left;
        direction: ltr;
    }
    
    /* Special styling for Arabic input */
    .arabic-input .stChatInput > div > div > input {
        text-align: right;
        direction: rtl;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"] {
        padding: 1rem;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom success/error/warning messages */
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
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #1e40af;
    }
    
    /* Input field styling */
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
    
    /* Health status indicators */
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
    # Add logo to the sidebar
    st.image("DIaBot Logo/logo.png", width=200)
    st.markdown("---")
    
    st.title("DiaBot AI")
    st.markdown("---")
    page = st.radio("Navigation", ["Health Analysis", "AI Health Assistant", "Health Education"])
    
    # Add Malaysian-specific resources
    st.markdown("---")
    st.markdown("### Malaysian Resources")
    st.markdown("- [Ministry of Health Malaysia](https://www.moh.gov.my/)")
    st.markdown("- [National Diabetes Institute (NADI)](http://www.nadi.org.my/)")
    st.markdown("- [Malaysian Diabetes Association](http://www.diabetes.org.my/)")
    
    # Add emergency contact information for Malaysia
    st.markdown("---")
    st.markdown("### Emergency Contacts (Malaysia)")
    st.markdown("**If you're experiencing a medical emergency, call 999 immediately.**")
    st.markdown("- Health Advisory: 03-8881 0200")
    st.markmarkdown("- Poison Control: 04-657 0099")
    st.markdown("- Mental Health: 03-7956 8145")

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

# Income category mapping (Malaysian Ringgit)
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
    """Convert exact age to age category used by the model"""
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
def predict_future_risk(current_features, current_age, years=5):
    """Predict diabetes risk for future years"""
    future_risks = []
    
    for year in range(years + 1):  # 0 to 5 years
        # Create a copy of current features
        future_features = current_features.copy()
        
        # Update age category for future year
        future_age = current_age + year
        future_age_category = get_age_category(future_age)
        future_features[18] = future_age_category  # Age is at index 18
        
        # Convert to tensor
        X_future = torch.tensor(future_features, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
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
                index=3  # Default to RM 3,000 to RM 4,000
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

    # Prediction buttons
    col1, col2 = st.columns(2)
    with col1:
        predict_current = st.button("Predict Current Risk", type="primary", use_container_width=True)
    with col2:
        predict_future = st.button("Predict Future Risk (5 Years)", type="secondary", use_container_width=True)

    # Current risk prediction
    if predict_current:
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            risk = probabilities[0][1].item()

        # Store health data in session state for the AI Health Assistant
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

        # Risk interpretation with custom styling
        if risk < 0.25:
            st.markdown('<div class="stSuccess">Low Risk – Maintain your healthy lifestyle.</div>', unsafe_allow_html=True)
        elif risk < 0.6:
            st.markdown('<div class="stWarning">Moderate Risk – Consider lifestyle improvements.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stError">High Risk – Please consult a healthcare professional.</div>', unsafe_allow_html=True)

        # Probability breakdown
        st.progress(risk)
        st.write("**Probability Breakdown:**")
        st.write(f"- No Diabetes: {(1-risk):.2%}")
        st.write(f"- Diabetes: {risk:.2%}")

        # Enhanced personalized insights with AI analysis
        st.subheader("Personalized Health Insights")
        
        # Create a personalized analysis based on the user's data
        insights = []
        
        # BMI analysis
        if BMI < 18.5:
            insights.append("Your BMI suggests you're underweight. Consider a balanced diet with adequate nutrition.")
        elif BMI >= 25 and BMI < 30:
            insights.append("Your BMI indicates overweight. Even a 5-7% weight loss can significantly reduce diabetes risk.")
        elif BMI >= 30:
            insights.append("Your BMI indicates obesity, a major diabetes risk factor. Focus on gradual weight loss through diet and exercise.")
        
        # Blood pressure analysis
        if HighBP == "Yes":
            insights.append("Managing your high blood pressure is crucial. Reduce sodium intake and monitor your levels regularly.")
        
        # Cholesterol analysis
        if HighChol == "Yes":
            insights.append("High cholesterol increases diabetes risk. Consider reducing saturated fats and increasing fiber intake.")
        
        # Activity analysis
        if PhysActivity == "No":
            insights.append("Regular physical activity (150 mins/week) can improve insulin sensitivity. Start with brisk walking.")
        
        # Diet analysis
        if Fruits == "No" or Veggies == "No":
            insights.append("Aim for 5 servings of fruits and vegetables daily. They're rich in fiber and antioxidants that protect against diabetes.")
        
        # Mental health analysis
        if MentHlth > 7:
            insights.append("Your mental health days are elevated. Stress management techniques may help reduce diabetes risk.")
        
        # Smoking analysis
        if Smoker == "Yes":
            insights.append("Smoking increases insulin resistance. Consider cessation programs to reduce your diabetes risk.")
        
        # Display the insights
        if insights:
            for i, insight in enumerate(insights, 1):
                st.write(f"{i}. {insight}")
        else:
            st.write("Your health profile shows no significant risk factors. Maintain your healthy habits!")
        
        # Add specific tips based on risk level
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
        
        # Add the required warning
        st.warning("**Important Notice:** These insights and recommendations are generated based on the information provided and are not a substitute for professional medical advice. Please consult with your healthcare provider for personalized medical guidance.")

    # Future risk prediction
    if predict_future:
        with st.spinner("Calculating future risk projections..."):
            # Predict risk for next 5 years
            future_risks = predict_future_risk(features, age, years=5)
            
            # Create data for visualization
            years = [f"Year {i}" for i in range(6)]  # Current year to year 5
            current_year = pd.Timestamp.now().year
            year_labels = [f"{current_year + i}" for i in range(6)]
            
            # Create DataFrame for chart
            risk_data = pd.DataFrame({
                "Year": years,
                "Year_Label": year_labels,
                "Risk": future_risks
            })
            
            # Display the chart
            st.subheader("5-Year Diabetes Risk Projection")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Line chart
                st.line_chart(risk_data, x="Year_Label", y="Risk")
            
            with col2:
                # Display the risk values
                st.write("**Risk by Year:**")
                for i, risk in enumerate(future_risks):
                    st.write(f"{year_labels[i]}: {risk:.2%}")
            
            # Add interpretation
            st.subheader("Future Risk Analysis")
            
            # Calculate risk change
            risk_change = future_risks[-1] - future_risks[0]
            
            if risk_change > 0.1:
                st.error(f"**Warning:** Your diabetes risk is projected to increase significantly by {year_labels[-1]} (+{risk_change:.2%}). Consider making lifestyle changes now to reduce this risk.")
            elif risk_change > 0.05:
                st.warning(f"**Notice:** Your diabetes risk is projected to increase by {year_labels[-1]} (+{risk_change:.2%}). Small lifestyle changes now can help mitigate this increase.")
            elif abs(risk_change) <= 0.05:
                st.info(f"**Stable:** Your diabetes risk is projected to remain relatively stable through {year_labels[-1]} ({risk_change:+.2%}).")
            else:
                st.success(f"**Improving:** Your diabetes risk is projected to decrease by {year_labels[-1]} ({risk_change:+.2%}). Keep up your healthy habits!")
            
            # Add recommendations based on future risk
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
            
            # Important disclaimer
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
    
    # Initialize the Bedrock client using secrets
    @st.cache_resource
    def get_bedrock_client():
        try:
            # Get credentials from Streamlit secrets
            aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
            aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
            region = st.secrets["AWS_DEFAULT_REGION"]
            
            # Initialize the client with credentials
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
    
    # Initialize session state for chat history with language tracking
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI health assistant specializing in diabetes care. Have you completed your health analysis yet? I can provide better advice if you share your health information with me.", "language": "English"}
        ]
    
    # Initialize language selection in session state
    if "language" not in st.session_state:
        st.session_state.language = "English"
    
    # Initialize show_quick_actions in session state
    if "show_quick_actions" not in st.session_state:
        st.session_state.show_quick_actions = True
    
    # Language selection dropdown
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
            st.rerun()
    
    # Apply Arabic input styling if Arabic is selected
    if st.session_state.language == "Arabic":
        st.markdown('<div class="arabic-input">', unsafe_allow_html=True)
    
    # Check if health data exists in session state
    health_data_exists = "health_data" in st.session_state
    
    # Display health data summary if available
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
    
    # Display quick actions if no messages beyond the initial one
    if st.session_state.show_quick_actions and len(st.session_state.messages) == 1:
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Get Diet Recommendations", help="Get personalized diet suggestions based on your health profile"):
                prompt = "Provide specific dietary recommendations for diabetes prevention"
                if health_data_exists:
                    prompt += f" for a {st.session_state.health_data.get('age')} year old {st.session_state.health_data.get('gender')} with a BMI of {st.session_state.health
