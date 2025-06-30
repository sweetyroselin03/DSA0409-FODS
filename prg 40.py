import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Name': ['Ronaldo', 'Messi', 'Mbappe', 'Neymar', 'Kane', 'Haaland', 'Modric', 'De Bruyne', 'Vinicius', 'Salah'],
    'Age': [38, 36, 25, 31, 30, 23, 38, 33, 23, 32],
    'Position': ['Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Midfielder', 'Midfielder', 'Forward', 'Forward'],
    'Goals': [820, 805, 255, 320, 280, 210, 120, 160, 130, 200],
    'Weekly_Salary($)': [500000, 600000, 450000, 490000, 400000, 420000, 300000, 350000, 380000, 370000]
}

df = pd.DataFrame(data)
df.to_csv('soccer_players.csv', index=False)
df = pd.read_csv('soccer_players.csv')

top_goal_scorers = df.sort_values(by='Goals', ascending=False).head(5)
top_paid_players = df.sort_values(by='Weekly_Salary($)', ascending=False).head(5)
average_age = df['Age'].mean()
above_avg_age_players = df[df['Age'] > average_age]

print("\nTop 5 Goal Scorers:\n", top_goal_scorers[['Name', 'Goals']])
print("\nTop 5 Highest Paid Players:\n", top_paid_players[['Name', 'Weekly_Salary($)']])
print(f"\nAverage Age of Players: {average_age:.2f}")
print("\nPlayers above average age:\n", above_avg_age_players[['Name', 'Age']])

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Position', palette='Set2')
plt.title("Distribution of Players by Position")
plt.ylabel("Number of Players")
plt.xlabel("Player Position")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Age', y='Weekly_Salary($)', hue='Position', palette='Set1', s=100)
plt.title("Age vs Salary of Players")
plt.xlabel("Age")
plt.ylabel("Weekly Salary ($)")
plt.tight_layout()
plt.show()
