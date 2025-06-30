import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Generate synthetic medical data
np.random.seed(42)
n = 300
data = pd.DataFrame({
    'Age': np.random.randint(30, 80, n),
    'Gender': np.random.choice([0, 1], n),  # 0=Female, 1=Male
    'BloodPressure': np.random.normal(120, 15, n),
    'Cholesterol': np.random.normal(200, 25, n)
})

# Label based on healthiness logic + noise
data['Outcome'] = np.where(
    (data['BloodPressure'] < 130) & (data['Cholesterol'] < 210) & (data['Age'] < 60),
    'Good', 'Bad'
)
# Add label noise
data['Outcome'] = np.where(np.random.rand(n) < 0.1,
                           np.where(data['Outcome'] == 'Good', 'Bad', 'Good'),
                           data['Outcome'])

# Step 2: Feature-label separation
X = data[['Age', 'Gender', 'BloodPressure', 'Cholesterol']]
y = data['Outcome']

# Show class balance
print("🔍 Outcome Class Distribution:\n", y.value_counts())

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train KNN classifier
k = 5  # you can tune this later
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Step 6: Evaluation Metrics
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2%}")

# Step 7: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=["Bad", "Good"])
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'🔍 Confusion Matrix (k={k})')
plt.tight_layout()
plt.show()

# Step 8: 2D Feature Scatter
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='BloodPressure', y='Cholesterol',
                hue='Outcome', style='Outcome', palette='Set2', s=80)
plt.title("📈 Blood Pressure vs Cholesterol by Outcome")
plt.grid(True)
plt.tight_layout()
plt.show()
