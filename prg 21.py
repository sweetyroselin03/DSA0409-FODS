import numpy as np
import pandas as pd
import scipy.stats as stats
import os

# Generate dataset if not found
file_name = "rare_elements.csv"
if not os.path.exists(file_name):
    np.random.seed(42)
    simulated_data = np.random.normal(loc=50, scale=10, size=1000)
    pd.DataFrame(simulated_data, columns=["Concentration"]).to_csv(file_name, index=False)
    print("Sample dataset 'rare_elements.csv' generated.")

# Load dataset
try:
    df = pd.read_csv(file_name)
    data = df.iloc[:, 0].dropna().values
except Exception as e:
    print("Error reading the file:", e)
    exit()

# Input with retry logic
def get_valid_input(prompt, input_type, condition=lambda x: True):
    while True:
        try:
            value = input_type(input(prompt))
            if not condition(value):
                raise ValueError
            return value
        except ValueError:
            print("Invalid input. Please try again.")

sample_size = get_valid_input("Enter sample size: ", int, lambda x: x > 0)
confidence_level = get_valid_input("Enter confidence level (e.g., 0.95 for 95%): ", float, lambda x: 0 < x < 1)
precision = get_valid_input("Enter desired level of precision (margin of error): ", float, lambda x: x > 0)

# Sampling
if sample_size > len(data):
    print("Sample size exceeds available data.")
    exit()

sample = np.random.choice(data, size=sample_size, replace=False)
mean_estimate = np.mean(sample)
std_error = stats.sem(sample)
t_score = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)

# Confidence Interval
margin_of_error = t_score * std_error
ci_lower = mean_estimate - margin_of_error
ci_upper = mean_estimate + margin_of_error

# Results
print(f"\n📌 Point Estimate (Sample Mean): {mean_estimate:.4f}")
print(f"📊 {int(confidence_level*100)}% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"📏 Calculated Margin of Error: ±{margin_of_error:.4f}")

if margin_of_error <= precision:
    print("✅ The estimate meets the desired level of precision.")
else:
    print("❌ The estimate does NOT meet the desired level of precision.")
    recommended_sample_size = int((t_score * np.std(data, ddof=1) / precision) ** 2)
    print(f"👉 Try increasing the sample size to approximately {recommended_sample_size} for desired precision.")
