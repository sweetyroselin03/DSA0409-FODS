import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Sample data for 18 adults (replace these values with your real data if needed)
data = {
    'Age': [23, 45, 34, 54, 29, 42, 33, 41, 37, 39, 28, 47, 52, 36, 40, 31, 44, 38],
    'FatPercent': [18.5, 25.2, 22.0, 28.3, 19.4, 24.8, 21.5, 23.7, 22.9, 23.3, 20.1, 27.2, 26.8, 21.9, 22.5, 20.8, 24.6, 22.7]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate statistics
print("Statistics for Age:")
print(f"Mean: {df['Age'].mean():.2f}")
print(f"Median: {df['Age'].median():.2f}")
print(f"Standard Deviation: {df['Age'].std():.2f}")

print("\nStatistics for Fat Percentage:")
print(f"Mean: {df['FatPercent'].mean():.2f}")
print(f"Median: {df['FatPercent'].median():.2f}")
print(f"Standard Deviation: {df['FatPercent'].std():.2f}")

# Boxplots for Age and Fat%
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(y='Age', data=df, color='skyblue')
plt.title("Boxplot of Age")

plt.subplot(1, 2, 2)
sns.boxplot(y='FatPercent', data=df, color='salmon')
plt.title("Boxplot of Body Fat Percentage")

plt.tight_layout()
plt.show()

# Scatter plot: Age vs Fat %
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Age', y='FatPercent', data=df, color='purple')
plt.title("Scatter Plot: Age vs Body Fat %")
plt.xlabel("Age")
plt.ylabel("Body Fat %")
plt.grid(True)
plt.show()

# Q-Q Plot for Age
plt.figure(figsize=(6, 4))
stats.probplot(df['Age'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Age")
plt.grid(True)
plt.show()

# Q-Q Plot for Fat Percentage
plt.figure(figsize=(6, 4))
stats.probplot(df['FatPercent'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Body Fat %")
plt.grid(True)
plt.show()
