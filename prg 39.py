import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Generate synthetic transaction data
np.random.seed(42)
df = pd.DataFrame({
    'CustomerID': range(1, 201),
    'AmountSpent': np.random.gamma(2, 150, 200).round(2),
    'ItemsPurchased': np.random.poisson(5, 200)
})

# Step 2: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['AmountSpent', 'ItemsPurchased']])

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 4: Cluster scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x='AmountSpent', y='ItemsPurchased', hue='Cluster',
    palette='Set2', s=100, edgecolor='black'
)
plt.title("🛍️ Customer Segmentation (Amount Spent vs Items Purchased)")
plt.xlabel("Total Amount Spent ($)")
plt.ylabel("Number of Items Purchased")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Cluster Centroids (in original scale)
centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_unscaled, columns=['AmountSpent', 'ItemsPurchased'])

# Plot again with centroids highlighted
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x='AmountSpent', y='ItemsPurchased', hue='Cluster',
    palette='Set2', s=80, edgecolor='gray'
)
plt.scatter(
    centroids_df['AmountSpent'], centroids_df['ItemsPurchased'],
    s=300, c='red', marker='X', label='Centroids'
)
for i, (x, y) in enumerate(centroids_unscaled):
    plt.text(x+2, y+0.2, f'C{i}', fontsize=9, fontweight='bold')

plt.title("📍 Customer Clusters with Centroids")
plt.xlabel("Total Amount Spent ($)")
plt.ylabel("Number of Items Purchased")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Cluster Summary
summary = df.groupby('Cluster')[['AmountSpent', 'ItemsPurchased']].mean().round(2)
summary['CustomerCount'] = df['Cluster'].value_counts().sort_index()
print("\n📊 Cluster Summary:\n", summary)

# (Optional) Step 7: Elbow Method to Find Optimal k
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaler.fit_transform(df[['AmountSpent', 'ItemsPurchased']]))
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertias, marker='o', color='purple')
plt.title('📐 Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.tight_layout()
plt.show()
