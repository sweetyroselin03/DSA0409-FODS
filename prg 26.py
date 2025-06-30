import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
try:
    df = pd.read_csv("housing_data.csv")  # CSV must contain 'Price' and other feature columns
except FileNotFoundError:
    print("Error: 'housing_data.csv' not found.")
    exit()

# Example: dataset has columns ['Area', 'Bedrooms', 'Location', 'Price']
# Separate features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Build a pipeline for preprocessing and regression
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X, y)

# Prompt user for new house features
print("\nEnter details of the house to predict price:")

new_data = {}
for col in X.columns:
    value = input(f"{col}: ")
    if col in numeric_cols:
        try:
            value = float(value)
        except ValueError:
            print(f"Invalid numeric value for {col}.")
            exit()
    new_data[col] = [value]

# Create DataFrame for the new input
new_house_df = pd.DataFrame(new_data)

# Predict the price
predicted_price = model.predict(new_house_df)[0]
print(f"\n🏠 Predicted House Price: ₹{predicted_price:,.2f}")
