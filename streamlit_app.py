import streamlit as st
import pandas as pd
from model import train_and_get_pipeline

# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="US Student Math Score Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance AI Predictor")
st.write("Predict a student's performance in **Mathematics** based on demographic data and other test scores.")

# --- Load Model & Data (Cached to run instantly after the first load) ---
@st.cache_resource
def get_model_and_data():
    model, raw_data = train_and_get_pipeline()
    return model, raw_data

try:
    with st.spinner("Initializing AI Engine and downloading dataset... Please wait..."):
        model, df = get_model_and_data()
    st.success("AI Model ready for predictions!")
except Exception as e:
    st.error(f"Error initializing app: {e}")
    st.stop()

# --- User Interface Input Form ---
st.header("📋 Enter Student Profile Data")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=df['gender'].unique())
    race_ethnicity = st.selectbox("Race/Ethnicity Group", options=sorted(df['race/ethnicity'].unique()))
    parent_education = st.selectbox("Parental Education Level", options=df['parental level of education'].unique())

with col2:
    lunch = st.selectbox("Lunch Type (US Demographics)", options=df['lunch'].unique(), 
                         help="'Standard' or 'Free/Reduced' (National School Lunch Program proxy)")
    test_prep = st.selectbox("Test Preparation Course", options=df['test preparation course'].unique())

st.subheader("📝 Existing Test Scores")
col3, col4 = st.columns(2)
with col3:
    reading_score = st.slider("Reading Score", min_value=0, max_value=100, value=70)
with col4:
    writing_score = st.slider("Writing Score", min_value=0, max_value=100, value=68)

# --- Prediction Processing ---
st.markdown("---")
if st.button("🔮 Predict Math Score", type="primary"):
    
    # Construct input dataframe matching exact schema expected by pipeline
    input_data = pd.DataFrame([{
        'gender': gender,
        'race/ethnicity': race_ethnicity,
        'parental level of education': parent_education,
        'lunch': lunch,
        'test preparation course': test_prep,
        'reading score': reading_score,
        'writing score': writing_score
    }])
    
    # Run Prediction
    prediction = model.predict(input_data)[0]
    
    # Display Results beautifully
    st.balloons()
    st.subheader("🎯 Prediction Result")
    
    # Color condition based on typical passing grade (60%)
    if prediction >= 60:
        st.success(f"The AI predicts this student will score: **{prediction:.1f} / 100** (Passing Status)")
    else:
        st.warning(f"The AI predicts this student will score: **{prediction:.1f} / 100** (Academic intervention recommended)")
        
    # Provide a breakdown context
    st.info(f"Context metrics: Student's literacy average is {((reading_score + writing_score)/2):.1f}%")