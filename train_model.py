# train_model.py
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


def train_and_evaluate():
  data_path = "climate_flood_data.csv"
  if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset {data_path} not found. Please ensure it is present."
    )

  df = pd.read_csv(data_path)
  df.columns = df.columns.str.strip()

  # Detect target column dynamically
  target_col = df.columns[-1]
  for col in df.columns:
    if "flood" in col.lower() or "risk" in col.lower():
      target_col = col
      break

  feature_cols = [
      "Rainfall_mm",
      "Temperature_C",
      "Humidity_pct",
      "River_Level_m",
      "Soil_Moisture_pct",
  ]

  X = df[feature_cols]
  y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

  # Stratified 80/20 split for reliable evaluation
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Train Random Forest with class_weight='balanced' to address false negatives
  model = RandomForestClassifier(
      n_estimators=100, random_state=42, class_weight="balanced"
  )
  model.fit(X_train, y_train)

  # Evaluate on held-out test set
  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(
      y_test, y_pred, average="binary", zero_division=0
  )
  cm = confusion_matrix(y_test, y_pred)

  # Extract False Negatives (Actual Positive [1], Predicted Negative [0])
  false_negatives = int(cm[1][0]) if cm.shape == (2, 2) else 0

  metrics = {
      "accuracy": float(acc),
      "precision": float(precision),
      "recall": float(recall),
      "f1_score": float(f1),
      "confusion_matrix": cm.tolist(),
      "false_negatives": false_negatives,
  }

  print("--- MODEL TRAINING COMPLETED ---")
  print(f"Test Accuracy: {acc * 100:.2f}%")
  print(classification_report(y_test, y_pred, zero_division=0))
  print(f"False Negatives (Missed Floods): {false_negatives}")

  # Save artifacts
  joblib.dump((model, feature_cols), "flood_model.pkl")
  with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
  print("Artifacts successfully saved: flood_model.pkl, metrics.json")


if __name__ == "__main__":
  train_and_evaluate()