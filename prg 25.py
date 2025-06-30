from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Accuracy on Test Data: {acc * 100:.2f}%")

# Input function with retry
def get_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

# Get user input
print("\n🔍 Enter flower measurements:")
sepal_length = get_input("Sepal length (cm): ")
sepal_width  = get_input("Sepal width (cm): ")
petal_length = get_input("Petal length (cm): ")
petal_width  = get_input("Petal width (cm): ")

# Prediction
new_flower = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(new_flower)
probabilities = model.predict_proba(new_flower)[0]
species = target_names[prediction[0]]

print(f"\n🌸 Predicted Species: **{species.capitalize()}**")
print("\n🔬 Confidence Scores:")
for i, prob in enumerate(probabilities):
    print(f"- {target_names[i].capitalize()}: {prob*100:.2f}%")

# Optional: Show decision tree
show_tree = input("\nDo you want to visualize the decision tree? (y/n): ").strip().lower()
if show_tree == "y":
    plt.figure(figsize=(12, 6))
    plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True, rounded=True)
    plt.title("🌳 Decision Tree Visualization")
    plt.tight_layout()
    plt.show()
