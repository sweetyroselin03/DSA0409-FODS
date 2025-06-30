import numpy as np

# Sample 3x3 matrix: each row = product, each column = sale record (e.g., prices)
sales_data = np.array([
    [100, 120, 110],   # Product 1
    [90, 95, 100],     # Product 2
    [130, 125, 135]    # Product 3
])

# Step 1: Calculate the overall average price
average_price = np.mean(sales_data)

# Display result
print(f"Average price of all products sold: ₹{average_price:.2f}")
