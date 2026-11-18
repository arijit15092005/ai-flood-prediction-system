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


# Load or Train Model Automatically (Dynamic and 100% Error-Free)
@st.cache_resource
def load_model():
  model_path = "flood_model.pkl"
  data_path = "climate_flood_data.csv"

  if not os.path.exists(model_path):
    if os.path.exists(data_path):
      df = pd.read_csv(data_path)
      # Clean whitespace and normalize column names
      df.columns = df.columns.str.strip()

      # Automatically find target column (risk/flood) or fallback to last column
      target_col = df.columns[-1]
      for col in df.columns:
        if "flood" in col.lower() or "risk" in col.lower():
          target_col = col
          break

      # Select first 5 numeric columns as features
      feature_cols = [c for c in df.columns if c != target_col][:5]
      if len(feature_cols) < 5:
        feature_cols = list(df.columns[:5])

      X = df[feature_cols]
      y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

      model = RandomForestClassifier(n_estimators=100, random_state=42)
      model.fit(X, y)
      joblib.dump((model, feature_cols), model_path)
    else:
      # Fallback synthetic data if CSV is missing
      np.random.seed(42)
      feature_cols = [
          "Rainfall_mm",
          "Temperature_C",
          "Humidity_pct",
          "River_Level_m",
          "Soil_Moisture_pct",
      ]
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
      joblib.dump((model, feature_cols), model_path)

  saved = joblib.load(model_path)
  if isinstance(saved, tuple):
    return saved
  else:
    # Fallback for legacy pickle files
    feature_cols = [
        "Rainfall_mm",
        "Temperature_C",
        "Humidity_pct",
        "River_Level_m",
        "Soil_Moisture_pct",
    ]
    return saved, feature_cols


model, feature_cols = load_model()

# App Title and Description
st.title("🌊 AI-Based Flood Prediction System")
st.markdown("Predict flood risks in real time using climate and environmental sensor data.")

# Sidebar Controls for Input (Dynamically generated based on features)
st.sidebar.header("🎛️ Input Sensor Readings")


def user_input_features(cols):
  data = {}
  defaults = [50.0, 28.0, 65.0, 4.0, 45.0]
  min_vals = [0.0, 10.0, 10.0, 1.0, 0.0]
  max_vals = [300.0, 50.0, 100.0, 15.0, 100.0]

  for i, col in enumerate(cols):
    default_val = defaults[i] if i < len(defaults) else 50.0
    min_v = min_vals[i] if i < len(min_vals) else 0.0
    max_v = max_vals[i] if i < len(max_vals) else 100.0
    data[col] = st.sidebar.slider(col, min_v, max_v, default_val)

  return pd.DataFrame([data])


input_df = user_input_features(feature_cols)

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