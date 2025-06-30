import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys

# Load dataset
try:
    df = pd.read_csv("car_data.csv")  # Replace with actual filename
except FileNotFoundError:
    print("❌ Error: 'car_data.csv' not found.")
    sys.exit()

if "Price" not in df.columns:
    print("❌ Error: Target column 'Price' is missing from the dataset.")
    sys.exit()

# Split features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Detect column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

# Fit the pipeline
pipeline.fit(X, y)

# --- Get user input ---
print("\n🚗 Enter details of the car you want to sell:")
new_data = {}
for col in X.columns:
    val = input(f"  ➤ {col}: ").strip()
    if col in numeric_cols:
        try:
            val = float(val)
        except ValueError:
            print(f"❌ Invalid input for '{col}'. It must be numeric.")
            sys.exit()
    new_data[col] = [val]

# Convert input to DataFrame
new_car_df = pd.DataFrame(new_data)

# Predict car price
predicted_price = pipeline.predict(new_car_df)[0]
print(f"\n💰 Estimated Car Price: ₹{predicted_price:,.2f}")

# --- Show Decision Tree Path (Text Format) ---
print("\n🧭 Simplified Decision Tree Path:")

# Refit tree on encoded data for readable output
tree = pipeline.named_steps["regressor"]
X_encoded = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

# Export tree as text
tree_description = export_text(tree, feature_names=list(feature_names), max_depth=3)
print(tree_description)

# --- Optional: Show top important features ---
importances = tree.feature_importances_
important_features = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n🌟 Top Influential Features (by importance):")
print(important_features.head(5).round(3).to_string())
