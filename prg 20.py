import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Customer ID': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010'],
    'Age': [25, 30, 22, 45, 36, 29, 50, 27, 33, 41],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Total Spending': [3200, 5400, 1800, 7500, 4000, 2200, 9100, 3100, 6600, 2700]
}

df = pd.DataFrame(data)

# Separate data by gender
male_data = df[df['Gender'] == 'Male'].sort_values(by='Age')
female_data = df[df['Gender'] == 'Female'].sort_values(by='Age')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(male_data['Age'], male_data['Total Spending'], marker='o', label='Male', color='blue')
plt.plot(female_data['Age'], female_data['Total Spending'], marker='o', label='Female', color='pink')

plt.title('Total Spending by Age and Gender')
plt.xlabel('Age')
plt.ylabel('Total Spending')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
