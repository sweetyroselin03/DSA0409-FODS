import matplotlib.pyplot as plt

# Data
months = ['Jan', 'Feb', 'Mar', 'Apr']
sales = [100, 120, 90, 150]

# Create 4 subplots in one figure
plt.figure(figsize=(16, 4))

# Line plot
plt.subplot(1, 4, 1)
plt.plot(months, sales, marker='o', color='blue')
plt.title('Line Plot')
plt.xlabel('Month')
plt.ylabel('Sales')

# Scatter plot
plt.subplot(1, 4, 2)
plt.scatter(months, sales, color='black')
plt.title('Scatter Plot')
plt.xlabel('Month')
plt.ylabel('Sales')

# Bar plot
plt.subplot(1, 4, 3)
plt.bar(months, sales, color='orange')
plt.title('Bar Plot')
plt.xlabel('Month')
plt.ylabel('Sales')

#Pie chart
plt.subplot(1, 4, 4)
plt.pie(sales, labels=months, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart')

# Adjust layout
plt.tight_layout()
plt.show()
