import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Generate sample transaction data
np.random.seed(42)
customer_ids = range(1, 201)
amount_spent = np.random.gamma(2, 150, 200).round(2)
items_purchased = np.random.poisson(5, 200)

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'AmountSpent': amount_spent,
    'ItemsPurchased': items_purchased
})

# Step 2: Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['AmountSpent', 'ItemsPurchased']])

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 4: Visualization - Scatter plot of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='AmountSpent', y='ItemsPurchased', hue='Cluster',
    data=df, palette='Set2', s=100, edgecolor='black'
)
plt.title("Customer Segmentation Based on Spending and Items Purchased")
plt.xlabel("Total Amount Spent ($)")
plt.ylabel("Number of Items Purchased")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Cluster Centroids (Unscaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=['AmountSpent', 'ItemsPurchased'])

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='AmountSpent', y='ItemsPurchased', hue='Cluster',
    data=df, palette='Set2', s=80, edgecolor='gray'
)
plt.scatter(
    centroid_df['AmountSpent'], centroid_df['ItemsPurchased'],
    s=250, c='red', marker='X', label='Centroids'
)
plt.title("Customer Clusters with Centroids")
plt.xlabel("Total Amount Spent ($)")
plt.ylabel("Number of Items Purchased")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Optional - Cluster Summary
cluster_summary = df.groupby('Cluster')[['AmountSpent', 'ItemsPurchased']].mean().round(2)
print("\nCluster-wise Summary:\n", cluster_summary)
