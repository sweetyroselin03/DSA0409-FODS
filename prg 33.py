import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Simulate car dataset
np.random.seed(42)
n = 200
engine_size = np.random.normal(2.0, 0.5, n)
horsepower = engine_size * np.random.uniform(60, 80) + np.random.normal(0, 10, n)
fuel_efficiency = 50 - engine_size * 5 + np.random.normal(0, 2, n)
weight = np.random.normal(1500, 300, n)
price = (engine_size * 7000 + horsepower * 50 - fuel_efficiency * 300 + weight * 2 +
         np.random.normal(0, 1000, n))

df = pd.DataFrame({
    'EngineSize': engine_size,
    'Horsepower': horsepower,
    'FuelEfficiency': fuel_efficiency,
    'Weight': weight,
    'Price': price
})

# Step 2: Select features and target
X = df[['EngineSize', 'Horsepower', 'FuelEfficiency', 'Weight']]
y = df['Price']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Model Evaluation:\nR² Score: {r2:.2f}\nMAE: {mae:.2f}\nMSE: {mse:.2f}")

# Step 6: Feature importance (coefficients)
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nFeature Influence on Price:\n", coeff_df)

# Step 7: Visualizations

# 1. Coefficient bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df, palette='coolwarm')
plt.title('Influence of Car Features on Price')
plt.tight_layout()
plt.show()

# 2. Actual vs Predicted prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
