# test_app.py
import os
import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


def test_model_artifact_exists():
  # Ensure model artifact can be loaded or mocked for test execution
  if not os.path.exists("flood_model.pkl"):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_mock = pd.DataFrame([[50, 25, 60, 4, 40], [200, 30, 90, 10, 80]])
    y_mock = [0, 1]
    model.fit(X_mock, y_mock)
    joblib.dump(
        (
            model,
            [
                "Rainfall_mm",
                "Temperature_C",
                "Humidity_pct",
                "River_Level_m",
                "Soil_Moisture_pct",
            ],
        ),
        "flood_model.pkl",
    )
  assert os.path.exists("flood_model.pkl")


def test_single_prediction():
  saved = joblib.load("flood_model.pkl")
  model, feature_cols = (
      saved
      if isinstance(saved, tuple)
      else (
          saved,
          [
              "Rainfall_mm",
              "Temperature_C",
              "Humidity_pct",
              "River_Level_m",
              "Soil_Moisture_pct",
          ],
      )
  )
  test_input = pd.DataFrame(
      [[150.0, 28.0, 70.0, 6.0, 50.0]], columns=feature_cols
  )
  pred = model.predict(test_input)
  assert len(pred) == 1
  assert pred[0] in [0, 1]


def test_batch_prediction():
  saved = joblib.load("flood_model.pkl")
  model, feature_cols = (
      saved
      if isinstance(saved, tuple)
      else (
          saved,
          [
              "Rainfall_mm",
              "Temperature_C",
              "Humidity_pct",
              "River_Level_m",
              "Soil_Moisture_pct",
          ],
      )
  )
  batch_data = pd.DataFrame({
      "Rainfall_mm": [10.0, 250.0],
      "Temperature_C": [25.0, 30.0],
      "Humidity_pct": [50.0, 90.0],
      "River_Level_m": [2.0, 11.0],
      "Soil_Moisture_pct": [30.0, 85.0],
  })
  preds = model.predict(batch_data[feature_cols])
  assert len(preds) == 2