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
    # Simulate weekend effect (lower admissions & ICU patients)
    is_weekend = (day % 7 in [5, 6])  # Saturday & Sunday

    # Generate hospital activity features
    total_admissions = np.random.randint(15, 50) if not is_weekend else np.random.randint(10, 35)
    total_discharges = np.random.randint(10, 45) if not is_weekend else np.random.randint(5, 25)
    icu_admissions = np.random.randint(2, 15) if not is_weekend else np.random.randint(1, 10)
    surgeries_today = np.random.randint(0, 10) if not is_weekend else np.random.randint(0, 5)
    
    bed_occupancy_rate = np.clip((prev_beds_occupied / 200), 0, 1)  # Assuming 200 total beds

    # Simulate seasonal demand (flu seasons, outbreaks)
    seasonal_factor = 5 * np.sin(2 * np.pi * day / 365)  # Yearly seasonality
    
    # Predict medical equipment needed (with some randomness)
    patient_monitors = max(5, int(icu_admissions * 1.2 + surgeries_today * 0.5 + seasonal_factor))
    defibrillators = max(2, int(surgeries_today * 0.3 + icu_admissions * 0.5 + seasonal_factor))
    infusion_pumps = max(8, int(icu_admissions * 1.5 + total_admissions * 0.3 + seasonal_factor))

    # Store data
    data.append([
        day, is_weekend, total_admissions, total_discharges, icu_admissions, surgeries_today,
        round(bed_occupancy_rate, 2),
        patient_monitors, defibrillators, infusion_pumps
    ])

    # Update for next iteration
    prev_beds_occupied = max(0, prev_beds_occupied + total_admissions - total_discharges)

# Create DataFrame
columns = [
    "Day_Index", "Is_Weekend", "Total_Admissions_Today", "Total_Discharges_Today",
    "ICU_Admissions_Today", "Surgeries_Today", "Bed_Occupancy_Rate",
    "Patient_Monitors_Required", "Defibrillators_Required", "Infusion_Pumps_Required"
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("equipment_data.csv", index=False)

# Display first few rows
print(df.head())
