import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv('climate_flood_data.csv')

# 2. Define features (X) and target (y)
X = df[['Rainfall_mm', 'Temperature_C', 'Humidity_pct', 'River_Level_m', 'Soil_Moisture_pct']]
y = df['Flood']

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate model performance
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model to disk
joblib.dump(model, 'flood_model.pkl')
print("Trained model saved successfully as 'flood_model.pkl'!")