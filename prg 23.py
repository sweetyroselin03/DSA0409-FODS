import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Load datasets
orders_df = pd.read_csv(r"C:\Users\trasr\Downloads\orders_data.csv", parse_dates=["Order Date"])
customers_df = pd.read_csv(r"C:\Users\trasr\Downloads\customer_info.csv")

# Merge datasets
merged_df = pd.merge(orders_df, customers_df, on="Customer ID")
merged_df.sort_values(by=["Customer ID", "Order Date"], inplace=True)

# Feature engineering
merged_df["Order Month"] = merged_df["Order Date"].dt.to_period("M")
merged_df["Day of Week"] = merged_df["Order Date"].dt.day_name()
merged_df["Order Gap"] = merged_df.groupby("Customer ID")["Order Date"].diff().dt.days
merged_df["Is Repeat"] = merged_df["Order Gap"].notnull()

# Summary statistics
total_orders = merged_df.shape[0]
unique_customers = merged_df["Customer ID"].nunique()
avg_gap = merged_df["Order Gap"].mean()
repeat_rate = merged_df["Is Repeat"].mean() * 100

# Display metrics
print("📊 Summary Metrics")
print(f"🧾 Total Orders: {total_orders}")
print(f"👥 Unique Customers: {unique_customers}")
print(f"⏱️ Average Days Between Orders: {avg_gap:.2f}")
print(f"🔁 Repeat Order Rate: {repeat_rate:.2f}%")

# Orders per month
monthly_orders = merged_df.groupby("Order Month")["Order ID"].count()

# Orders per weekday
weekday_orders = merged_df["Day of Week"].value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

# Frequent customers
orders_per_customer = merged_df["Customer ID"].value_counts()
frequent_customers = orders_per_customer[orders_per_customer > 5]
frequent_df = pd.DataFrame({
    "Customer ID": frequent_customers.index,
    "Order Count": frequent_customers.values
})

# Customer segments by order gap
customer_gap = merged_df.groupby("Customer ID")["Order Gap"].mean()
segments = pd.cut(customer_gap, bins=[0, 7, 30, 90, float("inf")],
                  labels=["Weekly", "Monthly", "Quarterly", "Rare"])
segment_counts = segments.value_counts().sort_index()

# --- Plots ---

# Plot 1: Monthly Orders
plt.figure(figsize=(10, 4))
monthly_orders.plot(marker="o", color="teal")
plt.title("📅 Orders Per Month")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_orders_trend.png")
plt.show()

# Plot 2: Weekday Orders
plt.figure(figsize=(8, 4))
weekday_orders.plot(kind="bar", color="salmon")
plt.title("📆 Orders by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("weekday_order_distribution.png")
plt.show()

# Plot 3: Customer Order Segments
plt.figure(figsize=(6, 6))
segment_counts.plot(kind="pie", autopct='%1.1f%%', colors=sns.color_palette("pastel"), ylabel="")
plt.title("📦 Customer Segmentation by Average Order Gap")
plt.tight_layout()
plt.savefig("customer_segments_pie.png")
plt.show()

# Optional: Save outputs
merged_df.to_csv("enhanced_merged_data.csv", index=False)
frequent_df.to_csv("frequent_customers.csv", index=False)
segment_counts.to_csv("customer_segments.csv")

print("\n✅ Enhanced data and plots saved successfully.")
