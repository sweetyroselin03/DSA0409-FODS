import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
df = pd.read_csv(r"C:\Users\balaj\Downloads\stock_data.csv", parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# Calculate key metrics
df['Daily_Change'] = df['Close'].diff()
df['Daily_Return(%)'] = df['Close'].pct_change() * 100
df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()

# Summary statistics
std_dev = df['Close'].std()
price_range = df['Close'].max() - df['Close'].min()
mean_price = df['Close'].mean()

print(f"📊 Stock Price Analysis:")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Price Range       : {price_range:.2f}")
print(f"Average Price     : {mean_price:.2f}")

# Set plot style
sns.set(style='whitegrid')

# Plot 1: Closing price over time
plt.figure(figsize=(12, 5))
sns.lineplot(x='Date', y='Close', data=df, color='blue')
plt.title('Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.tight_layout()
plt.show()

# Plot 2: Daily return distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Daily_Return(%)'].dropna(), bins=30, kde=True, color='orange')
plt.title('Distribution of Daily Return (%)')
plt.xlabel('Daily Return (%)')
plt.tight_layout()
plt.show()

# Plot 3: Boxplot of prices
plt.figure(figsize=(6, 4))
sns.boxplot(y='Close', data=df, color='lightgreen')
plt.title('Boxplot of Closing Prices')
plt.tight_layout()
plt.show()

# Plot 4: Rolling mean trend
plt.figure(figsize=(12, 5))
sns.lineplot(x='Date', y='Close', data=df, label='Close Price', color='gray')
sns.lineplot(x='Date', y='Rolling_Mean_20', data=df, label='20-Day Rolling Mean', color='red')
plt.title('Stock Price with 20-Day Rolling Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
