import pandas as pd
import scipy.stats as stats
import numpy as np

# Sample data (replace with actual blood pressure reductions)
drug_group = [12, 14, 10, 11, 15, 13, 16, 14, 13, 12,
              11, 13, 12, 14, 15, 16, 11, 12, 14, 13,
              13, 15, 16, 12, 14]  # 25 patients

placebo_group = [4, 5, 6, 3, 4, 5, 3, 4, 6, 5,
                 4, 3, 5, 4, 4, 5, 6, 3, 4, 5,
                 4, 5, 4, 3, 4]  # 25 patients

# Convert to Series
drug_series = pd.Series(drug_group)
placebo_series = pd.Series(placebo_group)

# Function to calculate confidence interval
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # standard error of the mean
    margin = stats.t.ppf((1 + confidence) / 2, df=n-1) * sem
    return (mean - margin, mean + margin)

# Calculate confidence intervals
drug_ci = confidence_interval(drug_series)
placebo_ci = confidence_interval(placebo_series)

# Print results
print(f"95% Confidence Interval for Drug Group: {drug_ci[0]:.2f} to {drug_ci[1]:.2f}")
print(f"95% Confidence Interval for Placebo Group: {placebo_ci[0]:.2f} to {placebo_ci[1]:.2f}")
