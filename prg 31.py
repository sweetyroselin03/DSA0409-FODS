import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Generate synthetic customer data
np.random.seed(42)
n = 300
df = pd.DataFrame({
    'CustomerID': range(1, n+1),
    'Age': np.random.randint(18, 60, n),
    'AnnualIncome': np.random.normal(60000, 15000, n).clip(20000, 150000),
    'BrowsingTime': np.random.normal(25, 10, n).clip(5, 60),  # in minutes per session
    'PurchaseFrequency': np.random.poisson(10, n),
    'SpendingScore': np.random.uniform(1, 100, n)
})

# Step 2: Feature selection and scaling
features = ['Age', 'AnnualIncome', 'BrowsingTime', 'PurchaseFrequency', 'SpendingScore']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Visualize clusters in 2D (using PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=80)
plt.title('Customer Segmentation (PCA View)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Pairplot (optional detailed view)
sns.pairplot(df[features + ['Cluster']], hue='Cluster', palette='Set2')
plt.suptitle("Pairplot of Features by Cluster", y=1.02)
plt.tight_layout()
plt.show()

# Step 6: Cluster Summary
summary = df.groupby('Cluster')[features].mean().round(2)
print("\nCluster Summary:\n", summary)

# Step 7: Heatmap of Cluster Centers
plt.figure(figsize=(8, 5))
sns.heatmap(summary, annot=True, fmt='.1f', cmap='YlGnBu')
plt.title('Cluster Feature Heatmap')
plt.tight_layout()
plt.show()
