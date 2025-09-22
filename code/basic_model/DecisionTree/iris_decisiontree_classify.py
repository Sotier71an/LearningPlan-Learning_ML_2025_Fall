from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# deviding dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# training decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# predict on test dataset
y_pred = model.predict(X_test)

# evaluate model
print(f"decision tree accuracy: {accuracy_score(y_test, y_pred):.2f}")
