import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
try:
    df = pd.read_csv("patient_data.csv")  # Replace with your dataset filename
except FileNotFoundError:
    print("Error: 'patient_data.csv' not found.")
    exit()

# Assume the last column is the target (0 or 1), and the rest are symptom features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (to improve model reliability)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ask user for value of k
try:
    k = int(input("Enter the number of neighbors (k): "))
    if k <= 0:
        raise ValueError
except ValueError:
    print("Invalid value for k. It must be a positive integer.")
    exit()

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Ask user for new patient symptom inputs
print(f"\nEnter {X.shape[1]} symptom values for a new patient:")

try:
    new_patient = []
    for col in X.columns:
        val = float(input(f"{col}: "))
        new_patient.append(val)
except ValueError:
    print("Invalid input. Please enter numeric values only.")
    exit()

# Scale and reshape the new patient data
new_patient_scaled = scaler.transform([new_patient])

# Make prediction
prediction = knn.predict(new_patient_scaled)
print("\nPrediction Result:")
print("✅ The patient has the condition." if prediction[0] == 1 else "❌ The patient does not have the condition.")
