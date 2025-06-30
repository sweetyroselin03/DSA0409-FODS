import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Step 1: Generate sample data
np.random.seed(1)
study_time = np.random.normal(5, 2, 100).clip(0)  # hours per day
exam_scores = (study_time * 10 + np.random.normal(0, 10, 100)).clip(0, 100)  # scores out of 100

df = pd.DataFrame({
    'StudyTime': study_time,
    'ExamScore': exam_scores
})

# Step 2: Correlation calculation
corr_value, _ = pearsonr(df['StudyTime'], df['ExamScore'])
print(f"Pearson Correlation Coefficient between study time and exam score: {corr_value:.2f}")

# Step 3: Scatter plot with regression line
plt.figure(figsize=(8, 5))
sns.regplot(x='StudyTime', y='ExamScore', data=df, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
plt.title('Study Time vs Exam Score with Regression Line')
plt.xlabel('Study Time (Hours)')
plt.ylabel('Exam Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Joint plot (scatter + histograms)
sns.jointplot(x='StudyTime', y='ExamScore', data=df, kind='reg', height=6, color='green')
plt.suptitle('Joint Distribution: Study Time and Exam Score', y=1.02)
plt.tight_layout()
plt.show()

# Step 5: Heatmap of correlation matrix
plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
