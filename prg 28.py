import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
try:
    df = pd.read_csv("customer_data.csv")  # Replace with actual file
except FileNotFoundError:
    print("Error: 'customer_data.csv' not found.")
    exit()

# Assume all columns are numeric features related to shopping behavior
X = df.select_dtypes(include=['float64', 'int64'])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ask user how many segments (clusters)
try:
    n_clusters = int(input("Enter the number of customer segments (clusters): "))
    if n_clusters <= 1:
        raise ValueError
except ValueError:
    print("Invalid input. Enter an integer greater than 1.")
    exit()

# Fit KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Get user input for new customer
print(f"\nEnter shopping-related features for a new customer:")

new_data = {}
for col in X.columns:
    try:
        value = float(input(f"{col}: "))
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        exit()
    new_data[col] = [value]

# Convert user input to DataFrame and scale
new_customer_df = pd.DataFrame(new_data)
new_customer_scaled = scaler.transform(new_customer_df)

# Predict cluster
cluster_label = kmeans.predict(new_customer_scaled)[0]
print(f"\n🧾 New customer belongs to Segment #{cluster_label}")
