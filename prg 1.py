import numpy as np

# Example: 4x4 matrix (each row is a student, each column is a subject)
# Columns: [Math, Science, English, History]
student_scores = np.array([
    [80, 90, 70, 85],
    [75, 88, 82, 79],
    [92, 95, 78, 88],
    [85, 87, 80, 90]
])

# Step 1: Calculate average for each subject (column-wise mean)
average_scores = np.mean(student_scores, axis=0)

# Step 2: Identify subject with the highest average score
subjects = ['Math', 'Science', 'English', 'History']
highest_avg_index = np.argmax(average_scores)
highest_avg_subject = subjects[highest_avg_index]

# Display results
print("Average scores by subject:")
for subj, avg in zip(subjects, average_scores):
    print(f"{subj}: {avg:.2f}")

print(f"\nSubject with highest average score: {highest_avg_subject} ({average_scores[highest_avg_index]:.2f})")
