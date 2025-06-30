import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

# Load dataset
try:
    df = pd.read_csv("customer_data.csv")  # Replace with actual file
except FileNotFoundError:
    print("❌ Error: 'customer_data.csv' not found.")
    sys.exit()

# Select numeric features only
X = df.select_dtypes(include=['float64', 'int64'])

if X.empty:
    print("❌ No numeric features found in the dataset.")
    sys.exit()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Get number of clusters from user with validation
while True:
    try:
        n_clusters = int(input("🔢 Enter the number of customer segments (clusters) [min: 2]: "))
        if n_clusters <= 1:
            raise ValueError
        break
    except ValueError:
        print("❗ Please enter a valid integer greater than 1.")

# Fit KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Optional: assign friendly segment labels
segment_names = {i: f"Segment {i+1}" for i in range(n_clusters)}

# Display basic centroid information
print("\n📊 Cluster centroids (standardized values):")
centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
print(centroids_df.round(2))

# Collect input for a new customer
print("\n✍️  Enter shopping-related features for a new customer:")

new_data = {}
for col in X.columns:
    while True:
        try:
            value = float(input(f"  ➤ {col}: "))
            new_data[col] = [value]
            break
        except ValueError:
            print("    ❗ Invalid input. Please enter a numeric value.")

# Create DataFrame and scale new input
new_customer_df = pd.DataFrame(new_data)
new_customer_scaled = scaler.transform(new_customer_df)

# Predict cluster
cluster_label = kmeans.predict(new_customer_scaled)[0]
segment = segment_names[cluster_label]

# Display result
print(f"\n✅ New customer has been classified into: **{segment}** (Cluster #{cluster_label})")

# Optional: Display similarity score to each cluster (euclidean distance)
distances = np.linalg.norm(kmeans.cluster_centers_ - new_customer_scaled, axis=1)
similarity_df = pd.DataFrame({'Segment': [segment_names[i] for i in range(n_clusters)],
                              'Distance': distances.round(3)}).sort_values(by='Distance')

print("\n📌 Similarity to all segments (lower is closer):")
print(similarity_df.to_string(index=False))
