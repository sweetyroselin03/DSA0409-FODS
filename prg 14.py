import pandas as pd

# Sample sales data (replace with your actual DataFrame)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7],
    'Age': [25, 34, 22, 34, 45, 22, 25],
    'PurchaseAmount': [120, 200, 150, 220, 300, 180, 130],
    'PurchaseDate': ['2025-06-01', '2025-06-03', '2025-06-05', '2025-06-10', '2025-06-15', '2025-06-18', '2025-06-25']
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert PurchaseDate to datetime
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

# Filter for the past month (e.g., June 2025)
df_last_month = df[df['PurchaseDate'].dt.month == 6]

# Frequency distribution of customer ages
age_frequency = df_last_month['Age'].value_counts().sort_index()

print("Frequency Distribution of Customer Ages:")
print(age_frequency)
