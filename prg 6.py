item_prices = [12.0, 8.0, 3.5]
quantities = [3, 2, 5]

discount_rate = 15  # in percent
tax_rate = 5        # in percent

# Step-by-step calculations
subtotals = [p * q for p, q in zip(item_prices, quantities)]
total_before_discount = sum(subtotals)

discount_amount = total_before_discount * (discount_rate / 100)
total_after_discount = total_before_discount - discount_amount

tax_amount = total_after_discount * (tax_rate / 100)
final_total = total_after_discount + tax_amount

# Output
print(f"Final Total: ${final_total:.2f}")
