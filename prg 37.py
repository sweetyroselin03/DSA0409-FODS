import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Step 1: Generate synthetic data
np.random.seed(1)
study_time = np.random.normal(5, 2, 100).clip(0)  # hours/day
exam_scores = (study_time * 10 + np.random.normal(0, 10, 100)).clip(0, 100)

df = pd.DataFrame({
    'StudyTime': study_time,
    'ExamScore': exam_scores
})

# Step 2: Correlation
corr_value, p_val = pearsonr(df['StudyTime'], df['ExamScore'])
print(f"📊 Pearson Correlation Coefficient: {corr_value:.2f}")
print(f"📉 P-value: {p_val:.4f}")

# Step 3: Scatter plot with regression line
plt.figure(figsize=(8, 5))
sns.regplot(
    x='StudyTime', y='ExamScore', data=df,
    scatter_kws={'color': 'navy', 'alpha': 0.6},
    line_kws={'color': 'red', 'label': f'y = 10x + noise\nr = {corr_value:.2f}'}
)
plt.title('📚 Study Time vs Exam Score (with Regression Line)')
plt.xlabel('Study Time (Hours)')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Jointplot
sns.jointplot(
    data=df, x='StudyTime', y='ExamScore',
    kind='reg', color='green', height=6,
    marginal_kws={'bins': 20, 'fill': True}
)
plt.suptitle('🔍 Joint Distribution: Study Time vs Exam Score', y=1.02)
plt.tight_layout()
plt.show()

# Step 5: Heatmap of correlation matrix
plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('📊 Correlation Heatmap')
plt.tight_layout()
plt.show()

# (Optional) Step 6: Outlier Detection
outliers = df[(df['StudyTime'] > 10) | (df['ExamScore'] < 30)]
if not outliers.empty:
    print("\n⚠️ Potential outliers detected:")
    print(outliers.round(2))
