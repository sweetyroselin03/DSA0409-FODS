import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create sample dataset
order_data = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C001', 'C003', 'C002', 'C004', 'C005', 'C001', 'C003', 'C005'],
    'order_date': ['2024-05-01', '2024-05-02', '2024-05-03', '2024-05-05', '2024-05-06',
                   '2024-05-07', '2024-05-08', '2024-05-10', '2024-05-11', '2024-05-12'],
    'product_name': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse',
                     'Laptop', 'Keyboard', 'Mouse', 'Keyboard', 'Laptop'],
    'order_quantity': [2, 1, 1, 3, 2, 1, 4, 2, 2, 1]
})

# Step 2: Save to CSV in current directory
order_data.to_csv('order_data.csv', index=False)

# Step 3: Read the dataset
df = pd.read_csv('order_data.csv')
df['order_date'] = pd.to_datetime(df['order_date'])

# Step 4: Analysis

# 1. Total number of orders made by each customer
orders_per_customer = df.groupby('customer_id').size()

# 2. Average order quantity for each product
avg_quantity_per_product = df.groupby('product_name')['order_quantity'].mean()

# 3. Earliest and latest order dates
earliest_date = df['order_date'].min()
latest_date = df['order_date'].max()

# Print Results
print("Total orders per customer:\n", orders_per_customer)
print("\n Average order quantity per product:\n", avg_quantity_per_product)
print(f"\n Earliest order date: {earliest_date.date()}")
print(f" Latest order date: {latest_date.date()}")

# Step 5: Visualization

# Bar chart: Total orders per customer
plt.figure(figsize=(6, 4))
orders_per_customer.plot(kind='bar', color='skyblue')
plt.title("Total Orders per Customer")
plt.xlabel("Customer ID")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

# Bar chart: Average quantity per product
plt.figure(figsize=(6, 4))
avg_quantity_per_product.plot(kind='bar', color='orange')
plt.title("Average Order Quantity per Product")
plt.xlabel("Product Name")
plt.ylabel("Average Quantity")
plt.tight_layout()
plt.show()
