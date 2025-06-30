import pandas as pd
import matplotlib.pyplot as plt

# Load and clean data
data = pd.read_csv(r"C:\Users\trasr\Downloads\monthly_weather_data.csv")
data.columns = data.columns.str.strip()

months = data['Month']
temp = data['Temperature(C)']
rain = data['Rainfall(mm)']

# Line Plot
plt.figure(figsize=(8, 4))
plt.plot(months, temp, label='Temperature (°C)', marker='o')
plt.plot(months, rain, label='Rainfall (mm)', marker='s')
plt.title('Monthly Temperature and Rainfall - Line Plot')
plt.xlabel('Month')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 4))
plt.scatter(months, temp, color='red', label='Temperature (°C)')
plt.scatter(months, rain, color='blue', label='Rainfall (mm)')
plt.title('Monthly Temperature and Rainfall - Scatter Plot')
plt.xlabel('Month')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Pie Chart for Rainfall Distribution
plt.figure(figsize=(6, 6))
plt.pie(rain, labels=months, autopct='%1.1f%%', startangle=90)
plt.title('Monthly Rainfall Distribution - Pie Chart')
plt.tight_layout()
plt.show()

