import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate sample temperature data (365 days for each city)
np.random.seed(0)
cities = ['Delhi', 'Mumbai', 'Chennai', 'Bangalore', 'Kolkata']
days = pd.date_range(start='2024-01-01', periods=365)

data = {
    'Date': np.tile(days, len(cities)),
    'City': np.repeat(cities, 365),
    'Temperature': np.concatenate([
        np.random.normal(25, 10, 365),  # Delhi
        np.random.normal(30, 5, 365),   # Mumbai
        np.random.normal(29, 6, 365),   # Chennai
        np.random.normal(27, 4, 365),   # Bangalore
        np.random.normal(28, 8, 365)    # Kolkata
    ])
}

df = pd.DataFrame(data)

# Step 2: Calculate statistics
city_stats = df.groupby('City')['Temperature'].agg(
    Mean_Temp='mean',
    Std_Dev='std',
    Max_Temp='max',
    Min_Temp='min'
)
city_stats['Temp_Range'] = city_stats['Max_Temp'] - city_stats['Min_Temp']

most_variable_city = city_stats['Temp_Range'].idxmax()
most_consistent_city = city_stats['Std_Dev'].idxmin()

print("City-wise Temperature Statistics:\n", city_stats.round(2))
print(f"\nCity with highest temperature range: {most_variable_city}")
print(f"City with most consistent temperature: {most_consistent_city}")

# Step 3: Visualizations

# Mean Temperature
plt.figure(figsize=(10, 5))
sns.barplot(x=city_stats.index, y=city_stats['Mean_Temp'], palette='coolwarm')
plt.title('Mean Temperature of Each City (°C)')
plt.ylabel('Mean Temperature (°C)')
plt.xlabel('City')
plt.grid(True)
plt.tight_layout()
plt.show()

# Standard Deviation
plt.figure(figsize=(10, 5))
sns.barplot(x=city_stats.index, y=city_stats['Std_Dev'], palette='viridis')
plt.title('Temperature Standard Deviation (Consistency)')
plt.ylabel('Standard Deviation (°C)')
plt.xlabel('City')
plt.grid(True)
plt.tight_layout()
plt.show()

# Line Plot of Daily Temperatures
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Temperature', hue='City', palette='tab10')
plt.title('Daily Temperature Trends by City')
plt.ylabel('Temperature (°C)')
plt.xlabel('Date')
plt.tight_layout()
plt.show()
