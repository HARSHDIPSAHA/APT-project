import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
df = pd.read_csv("equipment_data.csv")
df = df.apply(pd.to_numeric, errors='coerce')

# Create lag features
lag_days = 3
for col in ["Total_Discharges_Today", "ICU_Admissions_Today", "Surgeries_Today", "Bed_Occupancy_Rate"]:
    for lag in range(1, lag_days + 1):
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

df.dropna(inplace=True)

# Define feature columns
features = [
    'Total_Discharges_Today', 'ICU_Admissions_Today', 'Surgeries_Today', 'Bed_Occupancy_Rate'
] + [f"{col}_lag{lag}" for col in [
    "Total_Discharges_Today", "ICU_Admissions_Today", "Surgeries_Today", "Bed_Occupancy_Rate"
] for lag in range(1, lag_days + 1)]

targets = ["Patient_Monitors_Required", "Defibrillators_Required", "Infusion_Pumps_Required"]

# Define model file paths
model_paths = {
    "Patient_Monitors_Required": r"predbed/arima_Patient_Monitors_Required.pkl",
    "Defibrillators_Required": r"H:\APT-project\predbed\arima_Defibrillators_Required.pkl",
    "Infusion_Pumps_Required": r"H:\APT-project\predbed\arima_Infusion_Pumps_Required.pkl"
}

models = {target: joblib.load(path) for target, path in model_paths.items()}

def predict_equipment_requirements(models, latest_data):
    predictions = {}
    exog_df = latest_data[features]
    
    for target, model in models.items():
        try:
            forecast = model.get_forecast(steps=1, exog=exog_df)
            predictions[target] = forecast.predicted_mean.iloc[0]
        except Exception as e:
            print(f"Prediction error for {target}: {e}")
            predictions[target] = None
    
    return predictions

# Select latest data for prediction
latest_data = df.iloc[[-1]]

# Make predictions
predictions = predict_equipment_requirements(models, latest_data)

print("\nNext Day Equipment Predictions:")
for equip, value in predictions.items():
    if value is not None:
        print(f"{equip}: {value:.2f}")
    else:
        print(f"{equip}: Prediction failed.")

def update_model_with_actual(models, new_data):
    """
    Fine-tune ARIMA models with new actual values.
    """
    for target, model in models.items():
        try:
            # Extract new actual value
            new_actual_value = new_data[target].values[0]
            new_exog = new_data[features].values

            # Update model with new data (online learning)
            model = model.extend([new_actual_value], exog=new_exog)

            # Save updated model
            model_path = f"arima_{target.replace(' ', '_')}.pkl"
            joblib.dump(model, model_path)

            print(f"Updated model for {target} with actual value: {new_actual_value:.2f}")
        except Exception as e:
            print(f"Error updating model for {target}: {e}")

# Simulating new actual data (latest row)
latest_data = df.iloc[[-1]].copy()  # Assuming last row is the new actual value
update_model_with_actual(models, latest_data)
