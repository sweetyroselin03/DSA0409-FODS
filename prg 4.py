import numpy as np

sales_data = np.array([15000, 18000, 21000, 25000])

total_sales = np.sum(sales_data)

first_quarter = sales_data[0]
fourth_quarter = sales_data[3]
percentage_increase = ((fourth_quarter - first_quarter) / first_quarter) * 100

percentage_increase = round(percentage_increase, 3)

# Output
print("Total Sales for the Year: $", total_sales)
print("Percentage Increase from Q1 to Q4:", percentage_increase, "%")
