import numpy as np

# Example house_data array: [Bedrooms, Square Footage, Sale Price]
house_data = np.array([
    [3, 1500, 250000],
    [5, 2500, 400000],
    [4, 1800, 320000],
    [6, 3000, 500000],
    [5, 2400, 450000]
])

# Step 1: Filter rows with more than 4 bedrooms
houses_with_more_than_4_bedrooms = house_data[house_data[:, 0] > 4]

# Step 2: Extract sale prices of filtered houses (column index 2)
sale_prices = houses_with_more_than_4_bedrooms[:, 2]

# Step 3: Calculate average sale price
average_sale_price = np.mean(sale_prices)

# Display result
print(f"Average sale price of houses with more than 4 bedrooms: ₹{average_sale_price:.2f}")
