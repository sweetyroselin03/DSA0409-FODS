import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Sample monthly sales dataset
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December'],
    'Sales': [12000, 15000, 13500, 16000, 17000, 14500,
              15500, 16500, 14000, 17500, 18000, 19000]
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

# Step 3: Line plot of monthly sales
plt.figure(figsize=(10, 5))
plt.plot(df['Month'], df['Sales'], marker='o', linestyle='-', color='blue')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 4: Bar plot of monthly sales
plt.figure(figsize=(10, 5))
plt.bar(df['Month'], df['Sales'], color='green')
plt.title("Monthly Sales Comparison")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
