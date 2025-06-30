import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\trasr\Downloads\date_temperature_100_dataset.csv")

df['Date'] = pd.to_datetime(df['Date'])

monthly_avg = df.resample('ME', on='Date').mean(numeric_only=True)

monthly_avg.index = monthly_avg.index.strftime('%B')

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Temperature (°C)'], label='Daily Temperature', alpha=0.6)
plt.plot(df['Date'].dt.to_period('M').drop_duplicates().dt.to_timestamp(), 
         monthly_avg['Temperature (°C)'].values, 
         label='Monthly Avg Temperature', color='red', linewidth=2)
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Monthly Average Temperatures:\n")
print(monthly_avg)
