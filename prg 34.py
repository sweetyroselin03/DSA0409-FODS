import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Generate synthetic medical data
np.random.seed(42)
n = 300
data = pd.DataFrame({
    'Age': np.random.randint(30, 80, n),
    'Gender': np.random.choice([0, 1], n),  # 0=Female, 1=Male
    'BloodPressure': np.random.normal(120, 15, n),
    'Cholesterol': np.random.normal(200, 25, n)
})
# Generate outcome based on logic + noise
data['Outcome'] = np.where(
    (data['BloodPressure'] < 130) & (data['Cholesterol'] < 210) & (data['Age'] < 60),
    'Good', 'Bad'
)
data['Outcome'] = np.where(np.random.rand(n) < 0.1, 
                           np.where(data['Outcome'] == 'Good', 'Bad', 'Good'), 
                           data['Outcome'])

# Step 2: Prepare features and labels
X = data[['Age', 'Gender', 'BloodPressure', 'Cholesterol']]
y = data['Outcome']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Step 6: Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Step 8: Feature Visualization (optional)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='BloodPressure', y='Cholesterol', hue='Outcome', style='Outcome', palette='Set2')
plt.title("Blood Pressure vs Cholesterol by Outcome")
plt.grid(True)
plt.tight_layout()
plt.show()
