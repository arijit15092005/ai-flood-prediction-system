import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model
@st.cache_resource
def load_model():
    return joblib.load('flood_model.pkl')

model = load_model()

# Page configuration
st.set_page_config(page_title="AI Flood Prediction System", page_icon="🌊", layout="centered")

st.title("🌊 AI-Based Flood Prediction System")
st.markdown("Predict flood risks in real time using climate and environmental sensor data.")

st.sidebar.header("Climate & Sensor Inputs")

# Interactive Sliders for user input
rainfall = st.sidebar.slider("Rainfall (mm)", 10.0, 300.0, 100.0, 1.0)
river_level = st.sidebar.slider("River Water Level (m)", 1.0, 10.0, 4.0, 0.1)
soil_moisture = st.sidebar.slider("Soil Moisture (%)", 20.0, 90.0, 50.0, 0.5)
temperature = st.sidebar.slider("Temperature (°C)", 20.0, 42.0, 30.0, 0.5)
humidity = st.sidebar.slider("Humidity (%)", 40.0, 95.0, 65.0, 0.5)

# Create input dataframe for prediction
input_data = pd.DataFrame({
    'Rainfall_mm': [rainfall],
    'Temperature_C': [temperature],
    'Humidity_pct': [humidity],
    'River_Level_m': [river_level],
    'Soil_Moisture_pct': [soil_moisture]
})

st.subheader("Current Sensor Readings")
st.dataframe(input_data, use_container_width=True)

# Prediction button
if st.button("Predict Flood Risk", type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"🚨 **HIGH FLOOD RISK DETECTED!** (Risk Probability: {probability:.1f}%)")
        st.markdown("⚠️ *Action Required: Alert local authorities and initiate evacuation protocols.*")
    else:
        st.success(f"✅ **NORMAL CONDITIONS / LOW RISK** (Flood Probability: {probability:.1f}%)")
        st.markdown("👍 *Status: Environmental parameters are within safe thresholds.*")