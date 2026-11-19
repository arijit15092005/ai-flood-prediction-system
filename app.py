# app.py
import json
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI-Based Flood Prediction System", page_icon="🌊", layout="wide"
)


@st.cache_resource
def load_model_artifacts():
  model_path = "flood_model.pkl"
  metrics_path = "metrics.json"

  if not os.path.exists(model_path):
    return None, None, None

  saved = joblib.load(model_path)
  if isinstance(saved, tuple):
    model, feature_cols = saved
  else:
    model = saved
    feature_cols = [
        "Rainfall_mm",
        "Temperature_C",
        "Humidity_pct",
        "River_Level_m",
        "Soil_Moisture_pct",
    ]

  metrics = None
  if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
      metrics = json.load(f)

  return model, feature_cols, metrics


model, feature_cols, metrics = load_model_artifacts()

# App Header
st.title("🌊 AI-Based Flood Prediction System")
st.markdown(
    "**Cloud Computing Lab Project** | Academic Decision-Support Prototype"
)

if model is None:
  st.error(
      "⚠️ **Critical Error:** Model artifact `flood_model.pkl` not found. Please"
      " run `python train_model.py` locally to generate training artifacts"
      " before launching the app."
  )
  st.stop()

# Sidebar Manual Controls
st.sidebar.header("🎛️ Simulated Environmental Controls")
st.sidebar.markdown(
    "Adjust parameters below to simulate meteorological and hydrological"
    " conditions:"
)


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
  return pd.DataFrame([data])


input_df = user_input_features()

# Display Current Telemetry
st.subheader("📊 Current Simulated Telemetry")
st.dataframe(input_df, use_container_width=True)

# Parameter Distribution Chart
st.subheader("📈 Parameter Distribution Overview")
st.bar_chart(input_df.T)

# Prediction Logic
if st.button("🚨 Run Flood Risk Analysis", type="primary"):
  prediction = model.predict(input_df)
  proba = model.predict_proba(input_df)
  confidence = proba[0][prediction[0]] * 100

  st.subheader("🎯 Prediction & Decision Analysis")
  if prediction[0] == 1:
    st.warning(
        "⚠️ **Higher Flood-Risk Pattern Detected:** Model classification"
        " indicates conditions consistent with elevated flood risk."
    )
  else:
    st.success(
        "✅ **Lower Flood-Risk Pattern Detected:** Environmental parameters are"
        " within safe nominal thresholds."
    )

  st.metric(
      label="Uncalibrated Model Probability Estimate",
      value=f"{confidence:.2f}%",
  )
  st.progress(int(confidence))

  st.markdown(
      "> **Academic Disclaimer:** *This application is an educational"
      " decision-support prototype trained on a synthetic dataset. Its"
      " predictions are not official flood warnings and must not be used for"
      " real-world emergency or disaster-management decisions.*"
  )

# Sidebar Model Performance Panel
with st.sidebar.expander("📊 Model Evaluation Panel"):
  if metrics:
    st.write(f"**Test Accuracy:** {metrics['accuracy']*100:.2f}%")
    st.write(f"**Precision:** {metrics['precision']*100:.2f}%")
    st.write(f"**Recall (Flood Class):** {metrics['recall']*100:.2f}%")
    st.write(f"**F1-Score:** {metrics['f1_score']*100:.2f}%")
    st.write(f"**False Negatives:** {metrics['false_negatives']}")
    st.markdown("---")
    st.caption(
        "Metrics computed on an isolated 20% test split from synthetic"
        " demonstration data."
    )
  else:
    st.write("Evaluation metrics unavailable. Run `train_model.py`.")

# Batch CSV Prediction Section
st.sidebar.markdown("---")
st.sidebar.header("📂 Batch CSV Prediction")
uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV dataset", type=["csv"]
)

if uploaded_file is not None:
  st.markdown("---")
  st.subheader("📋 Batch CSV Prediction Results")
  try:
    batch_df = pd.read_csv(uploaded_file)
    batch_df.columns = batch_df.columns.str.strip()

    # Schema Validation
    missing_cols = [col for col in feature_cols if col not in batch_df.columns]
    if missing_cols:
      st.error(
          f"❌ Uploaded CSV is missing required feature columns:"
          f" {missing_cols}. Expected columns: {feature_cols}"
      )
    else:
      X_batch = batch_df[feature_cols]
      batch_preds = model.predict(X_batch)
      batch_df["Predicted_Status"] = [
          "Higher Flood Risk" if p == 1 else "Lower Flood Risk"
          for p in batch_preds
      ]
      st.dataframe(batch_df, use_container_width=True)

      csv_output = batch_df.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Download Batch Results as CSV",
          data=csv_output,
          file_name="flood_predictions_output.csv",
          mime="text/csv",
      )
      st.success("Batch prediction completed successfully!")
  except Exception as e:
    st.error(
        f"Error processing CSV file: {e}. Please ensure valid numerical data."
    )