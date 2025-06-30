import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Generate synthetic data
np.random.seed(42)
customer_ids = range(1, 201)
amount_spent = np.random.gamma(2, 150, 200).round(2)
visit_frequency = np.random.poisson(8, 200)

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'AmountSpent': amount_spent,
    'VisitFrequency': visit_frequency
})

# Step 2: Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['AmountSpent', 'VisitFrequency']])

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 4: Inverse transform centroids for visualization
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=['AmountSpent', 'VisitFrequency'])

# Step 5: Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='AmountSpent', y='VisitFrequency', hue='Cluster', palette='Set2', s=80, edgecolor='gray')
plt.scatter(
    centroid_df['AmountSpent'], centroid_df['VisitFrequency'],
    s=250, c='red', marker='X', label='Centroids'
)
plt.title('Customer Segments Based on Spending & Visit Frequency')
plt.xlabel('Total Amount Spent ($)')
plt.ylabel('Visit Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Optional summary of clusters
summary = df.groupby('Cluster')[['AmountSpent', 'VisitFrequency']].mean().round(2)
print("\nCluster Summary:\n", summary)
