import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic real estate data
np.random.seed(42)
n = 200
house_size = np.random.normal(1500, 400, n).clip(500, 3000)  # in sq ft
bedrooms = np.random.choice([2, 3, 4, 5], size=n, p=[0.2, 0.4, 0.3, 0.1])
location_index = np.random.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])  # 1=Urban, 2=Suburban, 3=Rural
price = (
    50000
    + house_size * 100
    + bedrooms * 10000
    + location_index * 15000
    + np.random.normal(0, 20000, n)
)

df = pd.DataFrame({
    'HouseSize': house_size,
    'Bedrooms': bedrooms,
    'LocationIndex': location_index,
    'Price': price
})

# Step 2: Bivariate Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='HouseSize', y='Price', data=df,
    hue='Bedrooms', size='LocationIndex',
    palette='viridis', sizes=(40, 120)
)
plt.title("📏 House Size vs. Price (Colored by Bedrooms, Sized by Location)")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 3: Train/Test Split using only 'HouseSize' as feature
X = df[['HouseSize']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Model Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n📊 Model Performance on Test Set:")
print(f"🔹 R² Score : {r2:.2f}")
print(f"🔹 MAE      : ${mae:,.2f}")
print(f"🔹 MSE      : {mse:,.2f}")
print(f"🔹 RMSE     : {rmse:,.2f}")

# Step 6: Plot Regression Line vs Actual
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['HouseSize'], y=y_test, label='Actual', color='blue')
sns.lineplot(x=X_test['HouseSize'], y=y_pred, label='Predicted', color='red')
plt.title("🔍 Actual vs Predicted House Prices (Linear Regression)")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='orange', bins=20)
plt.title("📉 Distribution of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Correlation Heatmap
plt.figure(figsize=(6, 4))
corr = df.corr().round(2)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("🧠 Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
