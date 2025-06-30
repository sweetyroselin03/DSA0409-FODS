import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier  # You can replace this with your model

# Load dataset
try:
    df = pd.read_csv("model_evaluation_data.csv")  # Replace with your dataset filename
except FileNotFoundError:
    print("Error: 'model_evaluation_data.csv' not found.")
    exit()

print("\nAvailable columns:", list(df.columns))

# Get user inputs
feature_input = input("\nEnter feature column names separated by commas (e.g., age,income,score): ")
target_input = input("Enter the name of the target column (e.g., churn): ")

feature_cols = [col.strip() for col in feature_input.split(',')]

# Check if input columns are valid
if not all(col in df.columns for col in feature_cols + [target_input]):
    print("❌ Invalid column names. Please check and try again.")
    exit()

# Prepare data
X = df[feature_cols]
y = df[target_input]

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (Random Forest as default, replace if needed)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
recall    = recall_score(y_test, y_pred, average='binary', zero_division=0)
f1        = f1_score(y_test, y_pred, average='binary', zero_division=0)

# Display metrics
print("\n📊 Model Evaluation Metrics:")
print(f"✅ Accuracy : {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall   : {recall:.4f}")
print(f"✅ F1-score : {f1:.4f}")
