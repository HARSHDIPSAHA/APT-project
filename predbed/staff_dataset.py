import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2 years of daily hospital data (730 days)
days = 730  
data = []

# Initial values
prev_beds_occupied = 120  # Starting occupied beds

for day in range(days):
    is_weekend = (day % 7 in [5, 6])  # Saturday & Sunday
    
    total_admissions = np.random.randint(15, 50) if not is_weekend else np.random.randint(10, 35)
    total_discharges = np.random.randint(10, 45) if not is_weekend else np.random.randint(5, 25)
    
    avg_los = np.random.uniform(3.5, 9.5)  # Average length of stay
    avg_age_admissions = np.random.randint(30, 75)  # Average age of admissions
    icu_patients = np.random.randint(2, 15) if not is_weekend else np.random.randint(1, 10)  # ICU patients
    surgeries_today = np.random.randint(0, 5) if not is_weekend else np.random.randint(0, 3)  # Surgeries
    
    total_beds_occupied = max(0, prev_beds_occupied + total_admissions - total_discharges)  
    
    # Seasonal factor (higher admissions in flu season)
    seasonal_factor = 10 * np.sin(2 * np.pi * day / 365)
    
    # Beds required tomorrow
    total_beds_required_tomorrow = max(0, total_beds_occupied + np.random.randint(-5, 10) + seasonal_factor)

    # Staff requirements based on hospital activity
    doctors_required = max(5, total_admissions // 5 + icu_patients // 2 + surgeries_today)  
    nurses_required = max(10, total_admissions // 2 + icu_patients + surgeries_today * 2)  
    technicians_required = max(3, surgeries_today + icu_patients // 3)  

    # Store data
    data.append([
        day, is_weekend, total_admissions, total_discharges, icu_patients, surgeries_today,
        total_beds_occupied, int(total_beds_required_tomorrow), 
        doctors_required, nurses_required, technicians_required
    ])

    # Update for next iteration
    prev_beds_occupied = total_beds_occupied

# Create DataFrame
columns = [
    "Day_Index", "Is_Weekend", "Total_Admissions_Today", "Total_Discharges_Today", 
    "ICU_Patients_Today", "Surgeries_Today", "Total_Beds_Occupied_Today", 
    "Total_Beds_Required_Tomorrow", "Doctors_Required_Tomorrow", 
    "Nurses_Required_Tomorrow", "Technicians_Required_Tomorrow"
]
df_staff = pd.DataFrame(data, columns=columns)

# Save to CSV
df_staff.to_csv("staff_data.csv", index=False)

# Display first few rows
print(df_staff.head())
