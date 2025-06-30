import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
try:
    df = pd.read_csv("churn_data.csv")  # Must include 'Churn' column (0 or 1)
except FileNotFoundError:
    print("Error: 'churn_data.csv' not found.")
    exit()

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Create preprocessing and logistic regression pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Standardizes the features
    ("logreg", LogisticRegression(solver='liblinear'))
])

# Train-test split for realism (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Prompt user for new customer input
print("\nEnter features for the new customer:")
new_data = {}

for col in X.columns:
    try:
        val = float(input(f"{col}: "))
    except ValueError:
        print(f"Invalid input for {col}. Please enter numeric values.")
        exit()
    new_data[col] = [val]

# Create a DataFrame from user input
new_customer = pd.DataFrame(new_data)

# Predict churn
prediction = pipeline.predict(new_customer)[0]
probability = pipeline.predict_proba(new_customer)[0][1]

# Show result
print(f"\n📊 Predicted Churn: {'Yes' if prediction == 1 else 'No'}")
print(f"🔍 Probability of Churn: {probability:.2%}")
