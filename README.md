# 🌊 AI-Based Flood Prediction System

## 🌐 Live Demo
- **Streamlit Community Cloud App:** https://ai-flood-prediction-system-3erjrn9zoa3gpfqpaxnoek.streamlit.app/
- **GitHub Repository:** https://github.com/arijit15092005/ai-flood-prediction-system

## 📋 Project Objective
The **AI-Based Flood Prediction System** is an educational, interactive machine learning web application built to evaluate meteorological and hydrological parameters and classify flood risk patterns. Developed as a third-year engineering cloud computing lab project, it demonstrates an end-to-end ML pipeline from data generation and training to version control and cloud deployment.

## 🛠️ Technology Stack
- **Language:** Python 3.11
- **Machine Learning:** Scikit-Learn (Random Forest Classifier with Class Weight Balancing)
- **Web Interface:** Streamlit Community Cloud
- **Version Control & CI/CD:** Git & GitHub
- **Testing:** Pytest

## 📐 System Architecture & Workflow
Simulated Manual Sliders / CSV Batch Upload
  → Data Validation & Schema Check
  → Feature Alignment
  → Random Forest Model Inference
  → Educational Risk Classification & Probability Estimate
  → Streamlit Dashboard Visualization

## 📊 Dataset & Limitations
- **Source:** Calibrated synthetic demonstration dataset (`climate_flood_data.csv`) generated via `generate_data.py`.
- **Generation Rules:** Labels are generated using transparent rule-based thresholds combining rainfall accumulation and river water levels.
- **Academic Disclaimer:** This is a synthetic prototype. High test accuracy reflects internal rule consistency rather than validated real-world meteorological forecasting accuracy. It must not be used for emergency flood warnings.

## 🤖 Machine Learning Model & Training (Viva Guide)
- **Algorithm:** Random Forest Classifier (`n_estimators=100`, `random_state=42`, `class_weight='balanced'`).
- **How it Works:** An ensemble classifier comprising multiple independent decision trees. Each tree votes on environmental feature thresholds, and majority voting determines the final classification. Class weighting ensures balanced sensitivity toward minority flood events, minimizing dangerous false negatives.

## 🚀 Local Execution Instructions
1. Navigate to your project folder in PowerShell:
   ```powershell
   cd C:\Users\lenovo\OneDrive\Desktop\flood-prediction-system