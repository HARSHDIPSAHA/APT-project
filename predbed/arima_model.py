import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("bed_data.csv")  # Ensure this dataset exists

# Ensure Date column is properly formatted
if 'Date' not in df.columns:
    df['Date'] = pd.date_range(start="2024-01-01", periods=len(df), freq='D')
else:
    df['Date'] = pd.to_datetime(df['Date'])  # Convert if already in CSV

df.set_index('Date', inplace=True)

# Selecting the target variable
y = df['Total_Beds_Occupied_Today']

# Function to perform ADF test
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Test p-value: {result[1]:.5f}')
    if result[1] > 0.05:
        print("Series is NOT stationary. Differencing needed.")
        return False
    else:
        print("Series is stationary. No differencing needed.")
        return True

# Step 1: Check Stationarity
if not adf_test(y):
    y = y.diff().dropna()  # First-order differencing
    adf_test(y)  # Recheck stationarity

# Step 2: Plot ACF & PACF to determine p and q
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(y, ax=ax[0], lags=20)
plot_pacf(y, ax=ax[1], lags=20)
plt.show()

# Set ARIMA parameters based on ACF/PACF
p, d, q = 1, 1, 1  # Adjust based on plots

# Step 3: Train-Test Split (80%-20%)
train_size = int(len(y) * 0.8)
train, test = y.iloc[:train_size], y.iloc[train_size:]

# Step 4: Train ARIMA Model
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

# Step 5: Forecast for the test period
forecast = model_fit.predict(start=len(train), end=len(y)-1)

# Step 6: Evaluate Performance
mae = mean_absolute_error(test, forecast)
print(f"Mean Absolute Error: {mae:.2f}")

# Step 7: Predict Next Day's Bed Requirement
next_day_forecast = model_fit.forecast(steps=1)
print(f"Predicted beds required tomorrow: {next_day_forecast[0]:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, forecast, label='Predicted', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Beds Occupied')
plt.title('Actual vs Predicted Bed Occupancy')
plt.legend()
plt.show()
