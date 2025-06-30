import pandas as pd

property_data = pd.DataFrame({
    'property_id': [101, 102, 103, 104, 105],
    'location': ['City Center', 'Suburb', 'Rural', 'Downtown', 'Suburb'],
    'bedrooms': [2, 4, 3, 5, 6],
    'area_sqft': [1100, 2100, 1600, 2800, 3200],
    'listing_price': [220000, 380000, 260000, 470000, 520000]
})

avg_price_by_location = property_data.groupby('location')['listing_price'].mean()
print("1. Average Listing Price by Location:")
print(avg_price_by_location)

num_properties_gt_4_bedrooms = property_data[property_data['bedrooms'] > 4].shape[0]
print("\n2. Number of Properties with More Than 4 Bedrooms:")
print(num_properties_gt_4_bedrooms)

largest_area_property = property_data.loc[property_data['area_sqft'].idxmax()]
print("\n3. Property with the Largest Area:")
print(largest_area_property)
import matplotlib.pyplot as plt

# Bar plot of average listing price by location
avg_price_by_location.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Average Listing Price by Location')
plt.xlabel('Location')
plt.ylabel('Average Price (₹)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

