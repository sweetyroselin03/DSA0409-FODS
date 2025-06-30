import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Sample sales dataset
data = {
    'order_date': ['2025-06-01', '2025-06-05', '2025-06-07', '2025-06-10', '2025-06-12',
                   '2025-06-15', '2025-06-18', '2025-06-20', '2025-06-22', '2025-06-25'],
    'product_name': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse',
                     'Monitor', 'Laptop', 'Keyboard', 'Monitor', 'Mouse'],
    'order_quantity': [2, 5, 1, 3, 2, 4, 1, 2, 3, 5]
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

# Step 3: Convert 'order_date' to datetime
df['order_date'] = pd.to_datetime(df['order_date'])

# Step 4: Filter data for the past 30 days
last_30_days = pd.Timestamp.today() - pd.Timedelta(days=30)
recent_sales = df[df['order_date'] >= last_30_days]

# Step 5: Group by product and sum quantities
product_sales = recent_sales.groupby('product_name')['order_quantity'].sum()

# Step 6: Get top 5 products
top_5_products = product_sales.sort_values(ascending=False).head(5)

# Step 7: Display results
print("📊 Top 5 Products Sold in the Past Month:")
print(top_5_products)

# Step 8: Plot the results
plt.figure(figsize=(6, 4))
top_5_products.plot(kind='bar', color='skyblue')
plt.title("Top 5 Products Sold in the Past Month")
plt.xlabel("Product Name")
plt.ylabel("Total Quantity Sold")
plt.tight_layout()
plt.show()
