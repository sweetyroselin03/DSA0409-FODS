import numpy as np
import pandas as pd
import scipy.stats as stats

# Load the dataset
try:
    df = pd.read_csv("rare_elements.csv")
    data = df.iloc[:, 0].dropna().values  # Assuming data is in the first column
except Exception as e:
    print("Error reading the file:", e)
    exit()

# Get user inputs
try:
    sample_size = int(input("Enter sample size: "))
    confidence_level = float(input("Enter confidence level (e.g., 0.95 for 95%): "))
    precision = float(input("Enter desired level of precision (margin of error): "))
except ValueError:
    print("Invalid input. Please enter numerical values.")
    exit()

# Randomly sample from the dataset
if sample_size > len(data):
    print("Sample size exceeds available data.")
    exit()

sample = np.random.choice(data, size=sample_size, replace=False)

# Point estimate
mean_estimate = np.mean(sample)

# Standard error
std_error = stats.sem(sample)

# t-score for given confidence level
t_score = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)

# Confidence interval
margin_of_error = t_score * std_error
ci_lower = mean_estimate - margin_of_error
ci_upper = mean_estimate + margin_of_error

# Display results
print(f"\nPoint Estimate (Sample Mean): {mean_estimate:.4f}")
print(f"{int(confidence_level*100)}% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"Calculated Margin of Error: ±{margin_of_error:.4f}")

# Precision check
if margin_of_error <= precision:
    print("✅ The estimate meets the desired level of precision.")
else:
    print("❌ The estimate does NOT meet the desired level of precision. Try increasing the sample size.")
