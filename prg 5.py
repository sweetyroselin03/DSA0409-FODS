import numpy as np

# Example: Fuel efficiency (in miles per gallon) of 5 car models
fuel_efficiency = np.array([22, 28, 35, 30, 25])

# Step 1: Calculate average fuel efficiency
average_efficiency = np.mean(fuel_efficiency)

# Step 2: Select two car models by index
# For example, compare model at index 0 and index 2
model1_eff = fuel_efficiency[0]  # e.g., 22 mpg
model2_eff = fuel_efficiency[2]  # e.g., 35 mpg

# Step 3: Calculate percentage improvement
# Formula: ((new - old) / old) * 100
percentage_improvement = ((model2_eff - model1_eff) / model1_eff) * 100

# Display results
print(f"Average fuel efficiency: {average_efficiency:.2f} mpg")
print(f"Percentage improvement from model 1 to model 2: {percentage_improvement:.2f}%")
