import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
orders_df = pd.read_csv(r"C:\Users\trasr\Downloads\orders_data.csv", parse_dates=["Order Date"])
customers_df = pd.read_csv(r"C:\Users\trasr\Downloads\customer_info.csv")

# Merge datasets
merged_df = pd.merge(orders_df, customers_df, on="Customer ID")
merged_df.sort_values(by=["Customer ID", "Order Date"], inplace=True)

# Add extra features
merged_df["Order Month"] = merged_df["Order Date"].dt.to_period("M")
merged_df["Day of Week"] = merged_df["Order Date"].dt.day_name()
merged_df["Order Gap"] = merged_df.groupby("Customer ID")["Order Date"].diff().dt.days
merged_df["Is Repeat"] = merged_df["Order Gap"].notnull()

# Summary metrics
total_orders = merged_df.shape[0]
unique_customers = merged_df["Customer ID"].nunique()
avg_gap = merged_df["Order Gap"].mean()
repeat_rate = merged_df["Is Repeat"].mean() * 100

print(f"Total Orders: {total_orders}")
print(f"Unique Customers: {unique_customers}")
print(f"Average Days Between Orders: {avg_gap:.2f}")
print(f"Repeat Order Rate: {repeat_rate:.2f}%")

# Orders per month
monthly_orders = merged_df.groupby("Order Month")["Order ID"].count()

# Orders per weekday
weekday_orders = merged_df["Day of Week"].value_counts().sort_index()

# Frequent customers
orders_per_customer = merged_df["Customer ID"].value_counts()
frequent_customers = orders_per_customer[orders_per_customer > 5]

# Customer segments by avg order gap
customer_gap = merged_df.groupby("Customer ID")["Order Gap"].mean()
segments = pd.cut(customer_gap, bins=[0, 7, 30, 90, float("inf")],
                  labels=["Weekly", "Monthly", "Quarterly", "Rare"])

# Plot 1: Monthly Order Trend
plt.figure(figsize=(10, 4))
monthly_orders.plot(kind="line", marker="o", title="Orders Per Month")
plt.ylabel("Number of Orders")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Weekday Order Distribution
plt.figure(figsize=(8, 4))
weekday_orders.plot(kind="bar", color="orange", title="Orders by Day of Week")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

# Save the enhanced data (optional)
merged_df.to_csv("enhanced_merged_data.csv", index=False)
