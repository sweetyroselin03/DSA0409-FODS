import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

# Use a nice Seaborn style
sns.set(style='whitegrid')

# Load the dataset
file_path = r"C:\Users\trasr\Downloads\date_temperature_100_dataset.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("File not found. Check the path.")
    exit()

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date (just in case)
df = df.sort_values(by='Date')

# Resample to get monthly average temperatures
monthly_avg = df.resample('M', on='Date').mean(numeric_only=True)

# Optional: Compute monthly max/min if available
monthly_stats = df.resample('M', on='Date').agg({
    'Temperature (°C)': ['mean', 'max', 'min']
})
monthly_stats.columns = ['Mean', 'Max', 'Min']

# Compute 7-day rolling average
df['7-Day Avg'] = df['Temperature (°C)'].rolling(window=7).mean()

# Plotting
plt.figure(figsize=(14, 7))

# Plot daily temperatures
plt.plot(df['Date'], df['Temperature (°C)'], label='Daily Temperature', alpha=0.5, color='gray')

# Plot 7-day moving average
plt.plot(df['Date'], df['7-Day Avg'], label='7-Day Rolling Avg', color='blue', linewidth=2)

# Plot monthly average
plt.plot(monthly_stats.index, monthly_stats['Mean'], label='Monthly Avg', color='red', linewidth=2, marker='o')

# Shade between monthly max and min if available
plt.fill_between(monthly_stats.index,
                 monthly_stats['Min'],
                 monthly_stats['Max'],
                 color='orange',
                 alpha=0.2,
                 label='Monthly Range (Min-Max)')

# Plot formatting
plt.title('🌡️ Temperature Trends Over Time', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Format the x-axis for better readability
date_format = DateFormatter("%b %Y")
plt.gca().xaxis.set_major_formatter(date_format)
plt.xticks(rotation=45)

# Save the figure
output_path = "enhanced_temperature_plot.png"
plt.savefig(output_path, dpi=300)
plt.show()

# Print Monthly Statistics
print("\n📊 Monthly Temperature Summary:\n")
monthly_stats.index = monthly_stats.index.strftime('%B')
print(monthly_stats.round(2))
