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
df['Rolling_Mean_50'] = df['Close'].rolling(window=50).mean()
df['Volatility'] = df['Daily_Return(%)'].rolling(window=10).std()

# Summary statistics
std_dev = df['Close'].std()
price_range = df['Close'].max() - df['Close'].min()
mean_price = df['Close'].mean()

print(f"\n📊 Stock Price Summary:")
print(f"🔹 Standard Deviation : {std_dev:.2f}")
print(f"🔹 Price Range         : {price_range:.2f}")
print(f"🔹 Average Closing Price: {mean_price:.2f}")

# Set plot style
sns.set(style='whitegrid')

# Plot 1: Closing price over time
plt.figure(figsize=(12, 5))
sns.lineplot(x='Date', y='Close', data=df, color='navy')
plt.title('📉 Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.tight_layout()
plt.show()

# Plot 2: Daily return histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['Daily_Return(%)'].dropna(), bins=30, kde=True, color='teal')
plt.title('📊 Distribution of Daily Return (%)')
plt.xlabel('Daily Return (%)')
plt.tight_layout()
plt.show()

# Plot 3: Boxplot of closing prices
plt.figure(figsize=(6, 4))
sns.boxplot(y='Close', data=df, color='lightblue')
plt.title('📦 Boxplot of Closing Prices')
plt.tight_layout()
plt.show()

# Plot 4: Close price and 20-day rolling average
plt.figure(figsize=(12, 5))
sns.lineplot(x='Date', y='Close', data=df, label='Close Price', color='gray')
sns.lineplot(x='Date', y='Rolling_Mean_20', data=df, label='20-Day MA', color='red')
sns.lineplot(x='Date', y='Rolling_Mean_50', data=df, label='50-Day MA', color='blue')
plt.title('📈 Stock Price with 20 & 50-Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 5: Daily change bar chart
plt.figure(figsize=(12, 4))
colors = df['Daily_Change'].apply(lambda x: 'green' if x > 0 else 'red')
plt.bar(df['Date'], df['Daily_Change'], color=colors, width=1)
plt.title('🔺 Daily Change in Closing Price')
plt.xlabel('Date')
plt.ylabel('Change')
plt.tight_layout()
plt.show()

# Plot 6: Volatility trend (optional)
plt.figure(figsize=(12, 4))
sns.lineplot(x='Date', y='Volatility', data=df, color='orange')
plt.title('📉 10-Day Rolling Volatility of Daily Returns')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.tight_layout()
plt.show()
