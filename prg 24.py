import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
try:
    df = pd.read_csv("patient_data.csv")  # Replace with actual path
except FileNotFoundError:
    print("❌ Error: 'patient_data.csv' not found.")
    exit()

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Ask user for value of k
while True:
    try:
        k = int(input("Enter the number of neighbors (k): "))
        if k <= 0:
            raise ValueError
        break
    except ValueError:
        print("Invalid value. Please enter a positive integer.")

# Train model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Evaluate model
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Accuracy on Test Data: {acc*100:.2f}%")
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# New patient prediction
print(f"\n🔍 Enter values for {X.shape[1]} symptoms to predict condition:")
new_patient = []
for col in X.columns:
    while True:
        try:
            val = float(input(f"{col}: "))
            new_patient.append(val)
            break
        except ValueError:
            print("Please enter a valid number.")

# Transform new data
new_patient_scaled = scaler.transform([new_patient])

# Predict
prediction = knn.predict(new_patient_scaled)[0]
classes = list(knn.classes_)

# Output
print("\n🩺 Prediction Result:")
if isinstance(prediction, str):
    print(f"✅ The predicted condition is: {prediction}")
elif prediction == 1:
    print("✅ The patient **has** the condition.")
else:
    print("❌ The patient **does NOT** have the condition.")
