import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
try:
    df = pd.read_csv("churn_data.csv")  # Ensure 'Churn' column exists
except FileNotFoundError:
    print("❌ Error: 'churn_data.csv' not found.")
    exit()

if "Churn" not in df.columns:
    print("❌ Dataset must contain a 'Churn' column as the target.")
    exit()

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Check for categorical columns (warn user)
if X.select_dtypes(include='object').shape[1] > 0:
    print("⚠️ Warning: Dataset contains categorical features. Please encode them or use OneHotEncoder.")
    exit()

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(solver='liblinear'))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Accuracy on Test Data: {acc:.2%}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# Collect user input
print("\n🧾 Enter customer details for churn prediction:")
new_data = {}

for col in X.columns:
    while True:
        try:
            val = float(input(f"{col}: "))
            new_data[col] = [val]
            break
        except ValueError:
            print(f"Please enter a valid numeric value for '{col}'.")

# Predict churn
new_customer = pd.DataFrame(new_data)
prediction = pipeline.predict(new_customer)[0]
probability = pipeline.predict_proba(new_customer)[0][1]

# Show result
print("\n📊 Prediction Result:")
print(f"🔁 Will Customer Churn? — {'✅ Yes' if prediction == 1 else '❌ No'}")
print(f"🔍 Churn Probability: {probability:.2%}")
