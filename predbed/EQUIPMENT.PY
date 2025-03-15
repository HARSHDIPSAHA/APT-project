import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("equipment_data.csv")
df = df.apply(pd.to_numeric, errors='coerce')

lag_days = 3
for col in ["Total_Discharges_Today", "ICU_Admissions_Today", "Surgeries_Today", "Bed_Occupancy_Rate"]:
    for lag in range(1, lag_days + 1):
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

df.dropna(inplace=True)

features = [
    'Total_Discharges_Today', 'ICU_Admissions_Today', 'Surgeries_Today', 'Bed_Occupancy_Rate'
] + [f"{col}_lag{lag}" for col in [
    "Total_Discharges_Today", "ICU_Admissions_Today", "Surgeries_Today", "Bed_Occupancy_Rate"
] for lag in range(1, lag_days + 1)]

targets = ["Patient_Monitors_Required", "Defibrillators_Required", "Infusion_Pumps_Required"]

def predict_equipment_requirements(models, latest_data):
    predictions = {}
    for target, path in models.items():
        model = joblib.load(path)
        exog_df = latest_data[features]
        forecast = model.get_forecast(steps=1, exog=exog_df)
        predictions[target] = forecast.predicted_mean.iloc[0]
    return predictions
models = {target: joblib.load(f"arima_{target.replace(' ', '_')}.pkl") for target in targets}

latest_data = df.iloc[[-1]]
predictions = predict_equipment_requirements(models, latest_data)
print("\nNext Day Equipment Predictions:")
for equip, value in predictions.items():
    print(f"{equip}: {value:.2f}")
