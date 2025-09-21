import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import boto3

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
    st.markdown("---")
    page = st.radio("Navigation", ["Health Analysis", "AI Health Assistant"])
    
    # Removed the Quick Stats section as requested

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
    5: "$25-35,000",
    6: "$35-50,000",
    7: "$50-75,000",
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
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI health assistant specializing in diabetes care. Have you completed your health analysis yet? I can provide better advice if you share your health information with me."}
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
            ["English", "Spanish", "French", "German", "Chinese", "Hindi", "Arabic", "Portuguese", "Russian", "Japanese"],
            index=0
        )
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm your AI health assistant specializing in diabetes care. Have you completed your health analysis yet? I can provide better advice if you share your health information with me."}
            ]
            st.session_state.show_quick_actions = True
            st.rerun()
    
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
                    prompt += f" for a {st.session_state.health_data.get('age')} year old {st.session_state.health_data.get('gender')} with a BMI of {st.session_state.health_data.get('bmi')}"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.show_quick_actions = False
                st.rerun()
        
        with col2:
            if st.button("Exercise Plan", help="Get a personalized exercise plan"):
                prompt = "Suggest an appropriate exercise routine"
                if health_data_exists:
                    activity_level = "active" if st.session_state.health_data.get('PhysActivity') == 'Yes' else "sedentary"
                    prompt += f" for someone who is currently {activity_level}"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.show_quick_actions = False
                st.rerun()
        
        with col3:
            if st.button("Risk Explanation", help="Understand your diabetes risk factors"):
                if health_data_exists:
                    prompt = f"Explain my diabetes risk of {st.session_state.health_data.get('risk')} and what factors contribute to it"
                else:
                    prompt = "What are the main risk factors for diabetes?"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.show_quick_actions = False
                st.rerun()
    
    # Display chat messages from history
    st.markdown("---")
    st.markdown("### Chat with Health Assistant")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Function to invoke Bedrock with Llama 3
    def invoke_llama(prompt, max_tokens=500, temperature=0.5):
        try:
            bedrock_client = get_bedrock_client()
            if bedrock_client is None:
                return "Error connecting to AI service. Please try again later."
            
            # Format the prompt for Llama 3 with language instruction
            language_instruction = f"Please respond in {st.session_state.language}." if st.session_state.language != "English" else ""
            
            formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
{language_instruction}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            
            # Prepare the request body for Llama 3
            body = json.dumps({
                "prompt": formatted_prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            })
            
            # Send the request to the Bedrock model
            response = bedrock_client.invoke_model(
                modelId='meta.llama3-8b-instruct-v1:0',
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            return response_body.get('generation', 'No response generated')
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Function to process user input and generate AI response
    def process_user_input(prompt):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create a context-aware prompt for the health assistant
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
                
                full_prompt = f"""
You are a friendly and knowledgeable health assistant specializing in diabetes prevention and management.
Provide helpful, evidence-based advice about nutrition, exercise, and lifestyle changes.
Always remind users to consult healthcare professionals for medical advice.

{health_context}
Current conversation context: {st.session_state.messages[-3:] if len(st.session_state.messages) > 3 else 'New conversation'}

User question: {prompt}

Please provide a helpful, concise response focused on diabetes prevention and management.
"""
                
                # Add language instruction if not English
                if st.session_state.language != "English":
                    full_prompt += f"\n\nPlease respond in {st.session_state.language}."
                
                full_response = invoke_llama(full_prompt)
                st.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Hide quick actions after first user input
        st.session_state.show_quick_actions = False
    
    # Check if we need to process a quick action prompt
    if len(st.session_state.messages) > 1 and st.session_state.messages[-1]["role"] == "user" and st.session_state.messages[-1]["content"] not in [msg["content"] for msg in st.session_state.messages[:-1]]:
        user_message = st.session_state.messages[-1]["content"]
        process_user_input(user_message)
        st.rerun()
    
    # Chat input
    chat_placeholder = st.empty()
    with chat_placeholder:
        prompt = st.chat_input("Ask about diabetes prevention, nutrition, or exercise...")
    
    if prompt:
        process_user_input(prompt)
        st.rerun()
