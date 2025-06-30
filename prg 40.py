import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Setup
data = {
    'Name': ['Ronaldo', 'Messi', 'Mbappe', 'Neymar', 'Kane', 'Haaland', 'Modric', 'De Bruyne', 'Vinicius', 'Salah'],
    'Age': [38, 36, 25, 31, 30, 23, 38, 33, 23, 32],
    'Position': ['Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Midfielder', 'Midfielder', 'Forward', 'Forward'],
    'Goals': [820, 805, 255, 320, 280, 210, 120, 160, 130, 200],
    'Weekly_Salary($)': [500000, 600000, 450000, 490000, 400000, 420000, 300000, 350000, 380000, 370000]
}

df = pd.DataFrame(data)

# Save and reload (mimic real-world workflow)
df.to_csv('soccer_players.csv', index=False)
df = pd.read_csv('soccer_players.csv')

# Key Insights
top_goal_scorers = df.sort_values(by='Goals', ascending=False).head(5)
top_paid_players = df.sort_values(by='Weekly_Salary($)', ascending=False).head(5)
average_age = df['Age'].mean()
above_avg_age_players = df[df['Age'] > average_age]

# Display Insights
print("\n⚽ Top 5 Goal Scorers:")
print(top_goal_scorers[['Name', 'Goals']])

print("\n💰 Top 5 Highest Paid Players:")
print(top_paid_players[['Name', 'Weekly_Salary($)']])

print(f"\n📊 Average Age of Players: {average_age:.2f} years")

print("\n🎯 Players Above Average Age:")
print(above_avg_age_players[['Name', 'Age']])

# Visualization 1: Distribution of Players by Position
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Position', palette='Set2')
plt.title("📌 Distribution of Players by Position")
plt.ylabel("Number of Players")
plt.xlabel("Position")
plt.tight_layout()
plt.show()

# Visualization 2: Age vs Salary
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Age', y='Weekly_Salary($)', hue='Position', palette='Set1', s=120)
plt.title("🧾 Age vs Weekly Salary of Players")
plt.xlabel("Age")
plt.ylabel("Weekly Salary ($)")
plt.tight_layout()
plt.show()

# Visualization 3: Top Goal Scorers
plt.figure(figsize=(8, 5))
sns.barplot(data=top_goal_scorers, x='Goals', y='Name', palette='Blues_r')
plt.title("🎯 Top 5 Goal Scorers")
plt.xlabel("Goals Scored")
plt.ylabel("Player")
plt.tight_layout()
plt.show()

# Visualization 4: Top Paid Players
plt.figure(figsize=(8, 5))
sns.barplot(data=top_paid_players, x='Weekly_Salary($)', y='Name', palette='Reds_r')
plt.title("💸 Top 5 Highest Paid Players")
plt.xlabel("Weekly Salary ($)")
plt.ylabel("Player")
plt.tight_layout()
plt.show()

# Visualization 5: Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], bins=7, kde=True, color='purple')
plt.title("📈 Age Distribution of Players")
plt.xlabel("Age")
plt.tight_layout()
plt.show()
