import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AI-Based Flood Prediction System", page_icon="🌊", layout="wide"
)


# Load or Train Model Automatically (Cloud-safe & Bulletproof)
@st.cache_resource
def load_model():
  model_path = "flood_model.pkl"
  data_path = "climate_flood_data.csv"

  if not os.path.exists(model_path):
    if os.path.exists(data_path):
      df = pd.read_csv(data_path)
      df.columns = df.columns.str.strip()
      X = df[[
          "Rainfall_mm",
          "Temperature_C",
          "Humidity_pct",
          "River_Level_m",
          "Soil_Moisture_pct",
      ]]
      y = df["Flood_Risk"]
      model = RandomForestClassifier(n_estimators=100, random_state=42)
      model.fit(X, y)
      joblib.dump(model, model_path)
    else:
      # Fallback synthetic dataset if CSV is missing
      np.random.seed(42)
      X = pd.DataFrame({
          "Rainfall_mm": np.random.uniform(0, 300, 100),
          "Temperature_C": np.random.uniform(15, 45, 100),
          "Humidity_pct": np.random.uniform(20, 100, 100),
          "River_Level_m": np.random.uniform(1, 10, 100),
          "Soil_Moisture_pct": np.random.uniform(10, 90, 100),
      })
      y = ((X["Rainfall_mm"] > 150) & (X["River_Level_m"] > 6)).astype(int)
      model = RandomForestClassifier(n_estimators=100, random_state=42)
      model.fit(X, y)
      joblib.dump(model, model_path)

  return joblib.load(model_path)


model = load_model()

# App Title and Description
st.title("🌊 AI-Based Flood Prediction System")
st.markdown("Predict flood risks in real time using climate and environmental sensor data.")

# Sidebar Controls for Input
st.sidebar.header("🎛️ Input Sensor Readings")


def user_input_features():
  rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 50.0)
  temperature = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 28.0)
  humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 65.0)
  river_level = st.sidebar.slider("River Water Level (m)", 1.0, 15.0, 4.0)
  soil_moisture = st.sidebar.slider("Soil Moisture (%)", 0.0, 100.0, 45.0)

  data = {
      "Rainfall_mm": rainfall,
      "Temperature_C": temperature,
      "Humidity_pct": humidity,
      "River_Level_m": river_level,
      "Soil_Moisture_pct": soil_moisture,
  }
  return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# Display Current Readings
st.subheader("📊 Current Sensor Readings")
st.dataframe(input_df, use_container_width=True)

# Prediction Button & Logic
if st.button("🚨 Predict Flood Risk"):
  if model is not None:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("🎯 Prediction Result")
    if prediction[0] == 1:
      st.error(
          "⚠️ **HIGH FLOOD RISK DETECTED!** Immediate precautions are advised."
      )
    else:
      st.success("✅ **NORMAL STATUS.** No immediate flood danger detected.")

    st.write(
        f"**Confidence Score:** {prediction_proba[0][prediction[0]] * 100:.2f}%"
    )
  else:
    st.error("Model could not be loaded.")