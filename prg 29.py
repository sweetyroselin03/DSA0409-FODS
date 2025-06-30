import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import sys

# Load dataset
try:
    df = pd.read_csv("model_evaluation_data.csv")  # Replace with actual file
except FileNotFoundError:
    print("❌ Error: 'model_evaluation_data.csv' not found.")
    sys.exit()

print("\n📄 Available columns in dataset:", list(df.columns))

# Get user input for features and target
feature_input = input("\n🛠️ Enter feature column names separated by commas (e.g., age,income,score): ")
target_input = input("🎯 Enter the name of the target column (e.g., churn): ")

feature_cols = [col.strip() for col in feature_input.split(',')]

# Validate column names
if not all(col in df.columns for col in feature_cols + [target_input]):
    print("❌ Error: One or more column names are invalid. Please check spelling.")
    sys.exit()

# Prepare data
X = df[feature_cols]
y = df[target_input]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Detect binary or multiclass classification
num_classes = len(set(y))
average_method = 'binary' if num_classes == 2 else 'macro'

# Compute metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
recall    = recall_score(y_test, y_pred, average=average_method, zero_division=0)
f1        = f1_score(y_test, y_pred, average=average_method, zero_division=0)

# Display results
print("\n📊 Model Evaluation Metrics:")
print(f"✅ Accuracy : {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f} ({'binary' if average_method == 'binary' else 'macro-average'})")
print(f"✅ Recall   : {recall:.4f}")
print(f"✅ F1-Score : {f1:.4f}")

# Optional: Full classification report
print("\n📝 Detailed Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
