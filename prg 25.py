from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split data (optional for realism)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Get user input for flower measurements
print("\nEnter flower measurements:")
try:
    sepal_length = float(input("Sepal length (cm): "))
    sepal_width  = float(input("Sepal width (cm): "))
    petal_length = float(input("Petal length (cm): "))
    petal_width  = float(input("Petal width (cm): "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

# Predict the species
new_flower = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(new_flower)
species = target_names[prediction[0]]

print(f"\n🌸 Predicted species: {species.capitalize()}")
