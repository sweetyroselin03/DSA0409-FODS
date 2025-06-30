import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Simulate car dataset
np.random.seed(42)
n = 200
engine_size = np.random.normal(2.0, 0.5, n)                         # liters
horsepower = engine_size * np.random.uniform(60, 80) + np.random.normal(0, 10, n)
fuel_efficiency = 50 - engine_size * 5 + np.random.normal(0, 2, n)  # mpg
weight = np.random.normal(1500, 300, n)                             # kg
price = (
    engine_size * 7000 +
    horsepower * 50 -
    fuel_efficiency * 300 +
    weight * 2 +
    np.random.normal(0, 1000, n)
)

df = pd.DataFrame({
    'EngineSize': engine_size,
    'Horsepower': horsepower,
    'FuelEfficiency': fuel_efficiency,
    'Weight': weight,
    'Price': price
})

# Step 2: Feature and target selection
X = df[['EngineSize', 'Horsepower', 'FuelEfficiency', 'Weight']]
y = df['Price']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n📊 Model Evaluation Metrics:")
print(f"🔹 R² Score : {r2:.3f}")
print(f"🔹 MAE      : ${mae:,.2f}")
print(f"🔹 MSE      : {mse:,.2f}")
print(f"🔹 RMSE     : ${rmse:,.2f}")

# Step 6: Feature importance (coefficients)
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'AbsoluteImpact': np.abs(model.coef_)
}).sort_values(by='AbsoluteImpact', ascending=False)

print("\n📌 Feature Influence on Car Price:\n", coeff_df[['Feature', 'Coefficient']])

# Step 7.1: Coefficient bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=coeff_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.title("📈 Influence of Car Features on Price")
plt.xlabel("Linear Coefficient")
plt.ylabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7.2: Actual vs Predicted Prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("🔍 Actual vs Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7.3: Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr().round(2), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("🔗 Feature Correlation Matrix")
plt.tight_layout()
plt.show()
