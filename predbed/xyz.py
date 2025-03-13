import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Load the data
df = pd.read_csv('bed_data.csv')

# === Feature Engineering ===
# 1. Critical: Add explicit trend feature to capture strong upward trend
df['trend'] = np.arange(len(df))

# 2. Create lag features for key metrics
for lag in [1, 2, 3, 7]:
    if len(df) > lag:
        df[f'beds_occupied_lag_{lag}'] = df['Total_Beds_Occupied_Today'].shift(lag)
        df[f'admissions_lag_{lag}'] = df['Total_Admissions_Today'].shift(lag)
        df[f'discharges_lag_{lag}'] = df['Total_Discharges_Today'].shift(lag)

# 3. Create rolling window features (moving averages)
for window in [3, 7, 14]:
    if len(df) > window:
        df[f'beds_occupied_ma_{window}'] = df['Total_Beds_Occupied_Today'].rolling(window=window).mean()
        df[f'admissions_ma_{window}'] = df['Total_Admissions_Today'].rolling(window=window).mean()
        df[f'discharges_ma_{window}'] = df['Total_Discharges_Today'].rolling(window=window).mean()

# 4. Create interaction features
df['net_flow'] = df['Total_Admissions_Today'] - df['Total_Discharges_Today']
df['los_x_admissions'] = df['Avg_LOS'] * df['Total_Admissions_Today']
df['age_x_los'] = df['Avg_Age_Admissions_Today'] * df['Avg_LOS'] / 10

# 5. Growth rate features
df['bed_growth_rate_1w'] = df['Total_Beds_Occupied_Today'].pct_change(periods=7)
df['bed_growth_rate_2w'] = df['Total_Beds_Occupied_Today'].pct_change(periods=14)

# Drop rows with NaN values from lag and rolling window features
df = df.dropna()

# === Model Training ===
# Prepare features and target
X = df.drop('Total_Beds_Required_Tomorrow', axis=1)
y = df['Total_Beds_Required_Tomorrow']

# Use time-based train-test split (last 20% for testing)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train XGBoost model with parameters tuned for time series
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42, early_stopping_rounds = 20
)

# Train with early stopping to prevent overfitting
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
   
    verbose=False
)

# === Evaluation ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f} beds")
print(f"RMSE: {rmse:.2f} beds")

# === Visualization ===
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
plt.title('Hospital Bed Predictions vs Actual')
plt.xlabel('Test Set Index')
plt.ylabel('Number of Beds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Save Model ===
joblib.dump(model, 'hospital_bed_prediction_model.joblib')

# === Prediction Function ===
# def predict_tomorrow_beds(new_data_df):
#     """Make a prediction for tomorrow's bed requirements"""
#     # Apply same feature engineering steps to new data
#     # This would require historical data access for lag features
#     processed_data = engineer_features(new_data_df)
#     return model.predict(processed_data)[0]
