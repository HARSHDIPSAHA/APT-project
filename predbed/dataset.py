import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2 years of daily hospital data (730 days)
days = 730  
data = []

# Initial values
prev_beds_occupied = 120  # Start with 120 occupied beds

for day in range(days):
    # Simulate weekend effect (lower discharges on weekends)
    is_weekend = (day % 7 in [5, 6])  # Saturday & Sunday
    
    total_admissions = np.random.randint(15, 50) if not is_weekend else np.random.randint(10, 35)
    total_discharges = np.random.randint(10, 45) if not is_weekend else np.random.randint(5, 25)
    
    avg_los = np.random.uniform(3.5, 9.5)  # Length of Stay
    avg_age_admissions = np.random.randint(30, 75)  # Average age
    total_beds_occupied = max(0, prev_beds_occupied + total_admissions - total_discharges)  
    
    # Trend: Slight increase over time
    seasonal_factor = 10 * np.sin(2 * np.pi * day / 365)  # Yearly seasonality (flu season, etc.)
    
    # Beds required tomorrow (simulate trend & variation)
    total_beds_required_tomorrow = max(0, total_beds_occupied + 
                                       np.random.randint(-5, 10) + seasonal_factor)

    # Store data
    data.append([
        day, is_weekend, total_admissions, total_discharges,
        round(avg_los, 2), avg_age_admissions, total_beds_occupied, 
        int(total_beds_required_tomorrow)
    ])

    # Update for next iteration
    prev_beds_occupied = total_beds_occupied

# Create DataFrame
columns = [
    "Day_Index", "Is_Weekend", "Total_Admissions_Today", 
    "Total_Discharges_Today", "Avg_LOS", "Avg_Age_Admissions_Today", 
    "Total_Beds_Occupied_Today", "Total_Beds_Required_Tomorrow"
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("bed_data.csv", index=False)

# Display first few rows
print(df.head())
