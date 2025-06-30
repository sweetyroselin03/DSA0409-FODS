import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load dataset
try:
    df = pd.read_csv("housing_data.csv")  # Must contain 'Price' and features
except FileNotFoundError:
    print("❌ Error: 'housing_data.csv' not found.")
    exit()

# Separate features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Train-test split to evaluate performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: preprocess + model
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n📈 Model R² Score on Test Set: {r2:.4f}")

# Get user input for prediction
print("\n🔍 Enter details of the house to predict price:")

def get_input(col):
    while True:
        val = input(f"{col}: ").strip()
        if col in numeric_cols:
            try:
                return float(val)
            except ValueError:
                print(f"⚠️ Please enter a valid number for {col}.")
        else:
            return val.title()  # Normalize categorical input

# Collect input for new prediction
new_data = {col: [get_input(col)] for col in X.columns}
new_house_df = pd.DataFrame(new_data)

# Predict price
predicted_price = model.predict(new_house_df)[0]
print(f"\n💰 Predicted House Price: ₹{predicted_price:,.2f}")
