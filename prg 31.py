import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Generate synthetic customer data
np.random.seed(42)
n = 300
df = pd.DataFrame({
    'CustomerID': range(1, n + 1),
    'Age': np.random.randint(18, 60, n),
    'AnnualIncome': np.random.normal(60000, 15000, n).clip(20000, 150000),
    'BrowsingTime': np.random.normal(25, 10, n).clip(5, 60),  # in minutes
    'PurchaseFrequency': np.random.poisson(10, n),
    'SpendingScore': np.random.uniform(1, 100, n)
})

# Step 2: Feature selection and scaling
features = ['Age', 'AnnualIncome', 'BrowsingTime', 'PurchaseFrequency', 'SpendingScore']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2.5: Elbow Method to determine optimal k
inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bo-')
plt.title("📉 Elbow Method For Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 3: Apply K-Means (k=4 as example from elbow)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Dimensionality reduction with PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Optional cluster labeling (can be changed based on summary)
cluster_labels = {
    0: "High Spenders",
    1: "Frequent Buyers",
    2: "Moderate Users",
    3: "Low Engagement"
}
df['Segment'] = df['Cluster'].map(cluster_labels)

# Consistent color palette
palette = sns.color_palette("Set2", n_colors=df['Cluster'].nunique())

# Step 5: PCA Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Segment', palette=palette, s=80)
plt.title('📊 Customer Segmentation (PCA View)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Pairplot
sns.pairplot(df[features + ['Segment']], hue='Segment', palette=palette)
plt.suptitle("🔍 Feature Relationships by Segment", y=1.02)
plt.tight_layout()
plt.show()

# Step 7: Cluster Summary Statistics
summary = df.groupby('Segment')[features].mean().round(2)
print("\n📌 Cluster Summary:\n")
print(summary)

# Step 8: Heatmap of Cluster Centers
plt.figure(figsize=(8, 5))
sns.heatmap(summary, annot=True, fmt='.1f', cmap='YlGnBu')
plt.title('🔥 Cluster Feature Heatmap')
plt.tight_layout()
plt.show()

# Step 9: Cluster Size Distribution
cluster_counts = df['Segment'].value_counts()
print("\n📊 Cluster Size Distribution:")
print(cluster_counts)

# Step 10: Silhouette Score
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"\n📈 Silhouette Score: {sil_score:.3f}")

# Optional: Save to CSV
# df.to_csv("segmented_customers.csv", index=False)
