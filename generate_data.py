import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1,000 samples of historical climate data
n_samples = 1000

rainfall = np.random.uniform(10, 300, n_samples)       # Rainfall in mm
temperature = np.random.uniform(20, 42, n_samples)    # Temperature in °C
humidity = np.random.uniform(40, 95, n_samples)       # Humidity in %
river_level = np.random.uniform(1.0, 10.0, n_samples) # River water level in meters
soil_moisture = np.random.uniform(20, 90, n_samples)  # Soil moisture in %

# Define a logical rule for flooding to generate realistic target labels (0 or 1)
flood_condition = (
    (rainfall > 180) & (river_level > 6.5) & (soil_moisture > 70)
) | (
    (rainfall > 220) & (river_level > 5.5)
)

flood_target = flood_condition.astype(int)

# Introduce slight noise / edge cases
noise_indices = np.random.choice(n_samples, size=50, replace=False)
flood_target[noise_indices] = 1 - flood_target[noise_indices]

# Create DataFrame
df = pd.DataFrame({
    'Rainfall_mm': rainfall,
    'Temperature_C': temperature,
    'Humidity_pct': humidity,
    'River_Level_m': river_level,
    'Soil_Moisture_pct': soil_moisture,
    'Flood': flood_target
})

# Save to CSV
df.to_csv('climate_flood_data.csv', index=False)
print("Dataset 'climate_flood_data.csv' generated successfully!")