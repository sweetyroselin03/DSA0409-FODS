import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load car dataset
try:
    df = pd.read_csv("car_data.csv")  # Must include target column 'Price'
except FileNotFoundError:
    print("❌ Error: 'car_data.csv' not found.")
    exit()

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Identify categorical and numeric features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Build pipeline with preprocessing and Decision Tree Regressor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

# Train model
pipeline.fit(X, y)

# Input for a new car
print("\n🚗 Enter the details of the car you want to sell:")

new_data = {}
for col in X.columns:
    value = input(f"{col}: ")
    if col in numeric_cols:
        try:
            value = float(value)
        except ValueError:
            print(f"❌ Invalid input for {col}. Must be a number.")
            exit()
    new_data[col] = [value]

# Create DataFrame for new input
new_car_df = pd.DataFrame(new_data)

# Predict price
predicted_price = pipeline.predict(new_car_df)[0]
print(f"\n💰 Predicted Price of the Car: ₹{predicted_price:,.2f}")

# Show decision path (textual format from raw tree)
# Fit tree separately to inspect internal structure
tree = pipeline.named_steps["regressor"]
X_encoded = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

# Display the decision path as text
print("\n🧭 Decision Tree Path for Prediction:\n")
tree_text = export_text(tree, feature_names=list(feature_names))
print(tree_text)
